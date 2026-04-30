import os
import time
import json
import re
import requests
from fastmcp import FastMCP
from dotenv import load_dotenv

mcp = FastMCP("PDBe MCP Server")

BASE_URL = "https://www.ebi.ac.uk/pdbe/api/pdb/entry"
_LAST_LLM_CALL_TS = 0.0


def normalize_pdb_id(pdb_id: str) -> str:
    pdb_id = pdb_id.strip().lower()
    if len(pdb_id) != 4 or not pdb_id.isalnum():
        raise ValueError("pdb_id must be a 4-character alphanumeric id (e.g. 1cbs)")
    return pdb_id


def call_llm(prompt: str, system_prompt: str | None = None, temperature: float = 0.2, max_tokens: int = 500) -> dict:
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
    request_timeout = int(os.getenv("LLM_REQUEST_TIMEOUT", "20"))
    request_timeout = max(5, min(request_timeout, 120))

    for attempt in range(1, max_retries + 1):
        response = requests.post(url, headers=headers, json=payload, timeout=request_timeout)
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
    return data


def _tool_schemas_for_llm() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_summary",
                "description": "Get summary details for a PDB entry by id.",
                "parameters": {
                    "type": "object",
                    "properties": {"pdb_id": {"type": "string"}},
                    "required": ["pdb_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_ligands",
                "description": "Get ligands present in a PDB entry.",
                "parameters": {
                    "type": "object",
                    "properties": {"pdb_id": {"type": "string"}},
                    "required": ["pdb_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_publications",
                "description": "Get publications for a PDB entry.",
                "parameters": {
                    "type": "object",
                    "properties": {"pdb_id": {"type": "string"}},
                    "required": ["pdb_id"],
                },
            },
        },
    ]


def _execute_local_tool(name: str, arguments: dict) -> dict | list:
    tool_map = {
        "get_summary": get_summary,
        "get_ligands": get_ligands,
        "get_publications": get_publications,
    }
    if name not in tool_map:
        raise ValueError(f"Unknown tool requested by LLM: {name}")
    return tool_map[name](**arguments)


def _pdbe_timeout_seconds() -> int:
    timeout = int(os.getenv("PDBE_REQUEST_TIMEOUT", "8"))
    return max(2, min(timeout, 30))


def _extract_pdb_id_from_question(question: str) -> str:
    match = re.search(r"\b([0-9][a-zA-Z0-9]{3})\b", question)
    if not match:
        raise ValueError("Could not detect PDB id in question (example: 1cbs)")
    return normalize_pdb_id(match.group(1))


@mcp.tool()
def healthcheck() -> dict:
    """Simple connectivity check."""
    return {"status": "ok", "service": "pdbe-mcp"}


@mcp.tool()
def get_summary(pdb_id: str) -> dict:
    """
    Get basic summary for a PDB entry (title, organism, method, resolution).
    """
    pdb_id = normalize_pdb_id(pdb_id)
    url = f"{BASE_URL}/summary/{pdb_id}"
    r = requests.get(url, timeout=_pdbe_timeout_seconds())
    r.raise_for_status()
    data = r.json().get(pdb_id, [])

    if not data:
        return {"error": "No data found"}

    entry = data[0]

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
    url = f"{BASE_URL}/ligand_monomers/{pdb_id}"
    r = requests.get(url, timeout=_pdbe_timeout_seconds())
    r.raise_for_status()
    data = r.json().get(pdb_id, [])

    ligands = []
    for lig in data:
        ligands.append({
            "chem_comp_id": lig.get("chem_comp_id"),
            "name": lig.get("chem_comp_name"),
            "formula": lig.get("chem_comp_formula"),
        })

    return ligands


@mcp.tool()
def get_publications(pdb_id: str) -> list:
    """
    Get primary publications associated with a PDB entry.
    """
    pdb_id = normalize_pdb_id(pdb_id)
    url = f"{BASE_URL}/publications/{pdb_id}"
    r = requests.get(url, timeout=_pdbe_timeout_seconds())
    r.raise_for_status()
    data = r.json().get(pdb_id, [])

    pubs = []
    for pub in data:
        pubs.append({
            "title": pub.get("title"),
            "journal": pub.get("journal_info", {}).get("journal"),
            "year": pub.get("year"),
            "doi": pub.get("doi"),
        })

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

    Example use:
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
    system_prompt: str = "unused-fast-mode",
    max_steps: int = 1,
) -> dict:
    """
    LLM agent that can call local PDBe tools before answering.
    """
    if not question.strip():
        raise ValueError("question must not be empty")

    # Fast deterministic mode for MCP clients with strict timeouts.
    steps = max(1, min(max_steps, 1))

    pdb_id = _extract_pdb_id_from_question(question)

    try:
        summary = get_summary(pdb_id)
        ligands = get_ligands(pdb_id)
    except Exception as exc:
        return {
            "model": "fallback-no-llm",
            "pdb_id": pdb_id,
            "steps_used": steps,
            "answer": (
                "1) Could not fetch PDBe data within timeout.\n"
                "2) Please retry in a moment or increase PDBE_REQUEST_TIMEOUT in .env.\n"
                f"3) Technical detail: {str(exc)}"
            ),
        }

    ligand_names = [lig.get("chem_comp_id") for lig in ligands[:8] if lig.get("chem_comp_id")]
    answer = (
        f"1) Summary for {pdb_id}: {summary.get('title', 'N/A')} "
        f"(method: {summary.get('experimental_method', 'N/A')}, resolution: {summary.get('resolution', 'N/A')}).\n"
        f"2) Ligands: {', '.join(ligand_names) if ligand_names else 'none reported'}.\n"
        f"3) Significance: this gives a quick structure-level view of chemistry and experimental quality for downstream analysis."
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
    system_prompt: str = "You are a senior structural biology assistant. Give a concise but insightful answer.",
) -> dict:
    """
    Slower but smarter mode: fetch PDBe context, then synthesize with LLM.
    """
    if not question.strip():
        raise ValueError("question must not be empty")

    pdb_id = _extract_pdb_id_from_question(question)
    summary = get_summary(pdb_id)
    ligands = get_ligands(pdb_id)
    publications = get_publications(pdb_id)

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
        max_tokens=350,
    )
    return {
        "model": llm_result.get("model"),
        "pdb_id": pdb_id,
        "answer": llm_result.get("response", ""),
        "usage": llm_result.get("usage", {}),
    }


def main() -> None:
    load_dotenv()
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8080"))

    if transport in ("http", "streamable-http"):
        # FastMCP streamable HTTP default path is /mcp.
        # Stateless mode avoids session crashes in some proxy/client reconnect flows.
        mcp.run(transport="streamable-http", host=host, port=port, stateless_http=True)
    elif transport == "sse":
        mcp.run(transport="sse", host=host, port=port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()