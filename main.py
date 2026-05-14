import json
import os
import re
import time
from typing import Any

import requests
from dotenv import load_dotenv
from fastmcp import FastMCP

mcp = FastMCP("PDBe MCP Server")

BASE_URL = os.getenv("PDBe_BASE_URL", "https://www.ebi.ac.uk/pdbe/api/pdb/entry").rstrip("/")
DEFAULT_USER_AGENT = os.getenv("PDBe_USER_AGENT", "pdbe-mcp/1.0")

_LAST_LLM_CALL_TS = 0.0
SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "application/json",
    }
)


def normalize_pdb_id(pdb_id: str) -> str:
    pdb_id = pdb_id.strip().lower()
    if len(pdb_id) != 4 or not pdb_id.isalnum():
        raise ValueError("pdb_id must be a 4-character alphanumeric id (e.g. 1cbs)")
    return pdb_id


def _pdbe_timeout_seconds() -> int:
    timeout = int(os.getenv("PDBE_REQUEST_TIMEOUT", "3"))
    return max(2, min(timeout, 10))


def _llm_timeout_seconds() -> int:
    timeout = int(os.getenv("LLM_REQUEST_TIMEOUT", "12"))
    return max(5, min(timeout, 60))


def _extract_pdb_id_from_question(question: str) -> str:
    match = re.search(r"\b([0-9][a-zA-Z0-9]{3})\b", question)
    if not match:
        raise ValueError(
            "No PDB id found. Include a 4-character PDB id like 1cbs in the question, "
            "or pass pdb_id explicitly."
        )
    return normalize_pdb_id(match.group(1))


def _pdbe_get(path: str) -> dict[str, Any]:
    url = f"{BASE_URL}/{path.lstrip('/')}"
    response = SESSION.get(url, timeout=_pdbe_timeout_seconds())
    response.raise_for_status()
    return response.json()


def _safe_pdbe_get(path: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return _pdbe_get(path), None
    except requests.Timeout:
        return None, "PDBe request timeout"
    except requests.RequestException as exc:
        return None, f"PDBe request failed: {exc}"
    except Exception as exc:
        return None, f"Unexpected PDBe error: {exc}"


def call_llm(prompt: str, system_prompt: str | None = None, temperature: float = 0.2, max_tokens: int = 500,) -> dict:
    """
    Call an OpenAI-compatible chat completion endpoint.

    Required env var:
      - LLM_API_KEY

    Optional env vars:
      - LLM_BASE_URL (default: https://api.openai.com/v1)
      - LLM_MODEL (default: gpt-4o-mini)
    """
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("LLM_API_KEY is not set")

    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    url = f"{base_url}/chat/completions"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    data = _post_llm_payload(url=url, headers=headers, payload=payload)

    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )

    return {
        "model": data.get("model", model),
        "response": content,
        "usage": data.get("usage", {}),
    }


def _post_llm_payload(url: str, headers: dict, payload: dict) -> dict:
    """Shared LLM POST with throttle and 429 retries."""
    global _LAST_LLM_CALL_TS

    min_interval_seconds = 2.0
    now = time.time()
    elapsed = now - _LAST_LLM_CALL_TS
    if elapsed < min_interval_seconds:
        time.sleep(min_interval_seconds - elapsed)

    max_retries = int(os.getenv("LLM_MAX_RETRIES", "1"))
    max_retries = max(1, min(max_retries, 5))
    backoff_seconds = 2
    request_timeout = _llm_timeout_seconds()

    data: dict[str, Any] | None = None

    for attempt in range(1, max_retries + 1):
        response = SESSION.post(url, headers=headers, json=payload, timeout=request_timeout)
        if response.status_code != 429:
            response.raise_for_status()
            data = response.json()
            break

        retry_after = response.headers.get("Retry-After")
        wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else backoff_seconds * attempt
        if attempt == max_retries:
            raise ValueError(
                "LLM provider returned 429 Too Many Requests. "
                "Check API billing/quota and rate limits, then retry later."
            )
        time.sleep(wait_seconds)

    _LAST_LLM_CALL_TS = time.time()
    return data or {}


@mcp.tool()
def healthcheck() -> dict:
    return {"status": "ok", "service": "pdbe-mcp"}


@mcp.tool()
def get_summary(pdb_id: str) -> dict:
    """
    Get basic summary for a PDB entry (title, organism, method, resolution).
    """
    pdb_id = normalize_pdb_id(pdb_id)
    data, error = _safe_pdbe_get(f"summary/{pdb_id}")
    if error:
        return {"pdb_id": pdb_id, "error": error}

    items = (data or {}).get(pdb_id, [])
    if not items:
        return {"pdb_id": pdb_id, "error": "No data found"}

    entry = items[0]
    return {
        "pdb_id": pdb_id,
        "title": entry.get("title"),
        "experimental_method": entry.get("experimental_method"),
        "resolution": entry.get("resolution"),
        "organism": entry.get("organism_scientific_name"),
    }


@mcp.tool()
def get_ligands(pdb_id: str) -> list:
    """
    Get ligands (small molecules) present in a PDB structure.
    """
    pdb_id = normalize_pdb_id(pdb_id)
    data, error = _safe_pdbe_get(f"ligand_monomers/{pdb_id}")
    if error:
        return [{"error": error}]

    items = (data or {}).get(pdb_id, [])
    ligands = []
    for lig in items:
        ligands.append(
            {
                "chem_comp_id": lig.get("chem_comp_id"),
                "name": lig.get("chem_comp_name"),
                "formula": lig.get("chem_comp_formula"),
            }
        )
    return ligands


@mcp.tool()
def get_publications(pdb_id: str) -> list:
    """
    Get primary publications associated with a PDB entry.
    """
    pdb_id = normalize_pdb_id(pdb_id)
    data, error = _safe_pdbe_get(f"publications/{pdb_id}")
    if error:
        return [{"error": error}]

    items = (data or {}).get(pdb_id, [])
    pubs = []
    for pub in items:
        pubs.append(
            {
                "title": pub.get("title"),
                "journal": pub.get("journal_info", {}).get("journal"),
                "year": pub.get("year"),
                "doi": pub.get("doi"),
            }
        )
    return pubs


@mcp.tool()
def ask_llm(
    prompt: str,
    system_prompt: str = "You are a helpful assistant for protein structure analysis.",
    temperature: float = 0.2,
    max_tokens: int = 500,
) -> dict:
    """
    Ask an LLM through an OpenAI-compatible API.

    Example:
      ask_llm("Explain what ligand binding site means in simple words")
    """
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    return call_llm(
        prompt=prompt.strip(),
        system_prompt=system_prompt.strip() if system_prompt else None,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@mcp.tool()
def ask_pdbe_agent(
    question: str,
    pdb_id: str | None = None,
    system_prompt: str = "unused-fast-mode",
    max_steps: int = 1,
) -> dict:
    """
    Fast deterministic mode for MCP clients with strict timeouts.
    """
    if not question.strip():
        raise ValueError("question must not be empty")

    steps = max(1, min(max_steps, 1))

    if pdb_id is None:
        pdb_id = _extract_pdb_id_from_question(question)
    else:
        pdb_id = normalize_pdb_id(pdb_id)

    summary = get_summary(pdb_id)
    ligands = get_ligands(pdb_id)

    if isinstance(summary, dict) and summary.get("error"):
        return {
            "model": "fallback-no-llm",
            "pdb_id": pdb_id,
            "steps_used": steps,
            "answer": f"1) Could not fetch summary for {pdb_id}.\n2) Technical detail: {summary.get('error')}",
        }

    if isinstance(ligands, list) and ligands and isinstance(ligands[0], dict) and ligands[0].get("error"):
        return {
            "model": "fallback-no-llm",
            "pdb_id": pdb_id,
            "steps_used": steps,
            "answer": f"1) Could not fetch ligands for {pdb_id}.\n2) Technical detail: {ligands[0].get('error')}",
        }

    ligand_names = [lig.get("chem_comp_id") for lig in ligands[:8] if lig.get("chem_comp_id")]

    answer = (
        f"1) Summary for {pdb_id}: {summary.get('title', 'N/A')} "
        f"(method: {summary.get('experimental_method', 'N/A')}, resolution: {summary.get('resolution', 'N/A')}).\n"
        f"2) Ligands: {', '.join(ligand_names) if ligand_names else 'none reported'}.\n"
        f"3) Significance: this gives a quick structure-level view of chemistry and experimental quality."
    )

    return {
        "model": "deterministic-fast",
        "pdb_id": pdb_id,
        "steps_used": steps,
        "answer": answer,
    }


@mcp.tool()
def ask_pdbe_agent_llm(
    question: str,
    pdb_id: str | None = None,
    system_prompt: str = "You are a senior structural biology assistant. Give a concise but insightful answer.",
    include_publications: bool = False,
) -> dict:
    """
    Slower but smarter mode: fetch PDBe context, then synthesize with LLM.
    """
    if not question.strip():
        raise ValueError("question must not be empty")

    if pdb_id is None:
        pdb_id = _extract_pdb_id_from_question(question)
    else:
        pdb_id = normalize_pdb_id(pdb_id)

    summary = get_summary(pdb_id)
    ligands = get_ligands(pdb_id)
    publications = get_publications(pdb_id) if include_publications else []

    prompt = (
        f"User question: {question.strip()}\n\n"
        f"PDB ID: {pdb_id}\n"
        f"Summary JSON: {json.dumps(summary, ensure_ascii=True)}\n"
        f"Ligands JSON: {json.dumps(ligands, ensure_ascii=True)}\n"
        f"Publications JSON: {json.dumps(publications, ensure_ascii=True)}\n\n"
        "Instructions:\n"
        "1) Answer in 3 numbered steps.\n"
        "2) Mention key ligand(s), method, and any resolution info.\n"
        "3) End with one short practical interpretation sentence.\n"
    )

    llm_result = call_llm(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=0.1,
        max_tokens=250,
    )

    return {
        "model": llm_result.get("model"),
        "pdb_id": pdb_id,
        "answer": llm_result.get("response", ""),
        "usage": llm_result.get("usage", {}),
    }


def main() -> None:
    load_dotenv()

    transport = os.getenv("MCP_TRANSPORT", "stdio").strip().lower()
    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8080"))

    if transport in ("http", "streamable-http"):
        mcp.run(transport="streamable-http", host=host, port=port, stateless_http=True)
    elif transport == "sse":
        mcp.run(transport="sse", host=host, port=port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()