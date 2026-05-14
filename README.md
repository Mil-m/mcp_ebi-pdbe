# mcp_building_pdbe

Starter template for a FastMCP 3.x server.

## What is included

- `healthcheck` tool: confirms the server is reachable.
- `lookup_entry` tool: placeholder tool that validates a PDB ID.
- `main.py`: runnable FastMCP server entrypoint.

## Run locally

### MCP inspector:

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
Arguments: /absolute/path/to/main.py<br>

OR

Settings:<br>
Transport Type: Streamable HTTP<br>
URL: http://localhost:8080/mcp
<br>
Arguments: /absolute/path/to/main.py<br>

### Server (for Streamable HTTP MCP inspector launch):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
MCP_TRANSPORT=streamable-http python main.py
```

## Usage example
(question) "What ligands are present in 1cbs?"
<br>or<br>
(question) "What ligands are present?"
(pdb_id) "1cbs"
