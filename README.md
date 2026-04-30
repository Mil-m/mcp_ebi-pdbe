# mcp_building_pdbe

Starter template for a FastMCP 3.x server.

## What is included

- `healthcheck` tool: confirms the server is reachable.
- `lookup_entry` tool: placeholder tool that validates a PDB ID.
- `main.py`: runnable FastMCP server entrypoint.

## Run locally

MCP inspector:

```bash
nvm install 20
nvm use 20
node -v
npx @modelcontextprotocol/inspector
```

http://127.0.0.1:6274

Settings:<br>
Transport Type: STDIO<br>
Command: /usr/local/bin/python3<br>
Arguments: main.py<br>

OR

```bash
python3 -m venv .venv   
source .venv/bin/activate
MCP_TRANSPORT=http python main.py
```

Settings:<br>
Transport Type: Streamable HTTP<br>
URL: http://localhost:8080/mcp
<br>
Arguments: main.py<br>

```bash
uv sync
uv run python main.py
```

If you use your existing virtual environment:

```bash
python3 -m venv .venv
pip install -r requirements.txt
python3 main.py
```

## Next steps

1. Replace `lookup_entry` in `main.py` with real PDBe API calls.
2. Add more MCP tools/resources/prompts for your workflow.
3. Connect this server to your MCP client config.
