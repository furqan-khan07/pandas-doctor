# Runs the MCP server over stdio (used by Glama and other MCP hosts).
FROM python:3.13-slim
RUN pip install --no-cache-dir "pandas-doctor[mcp]==0.1.1"
ENTRYPOINT ["pandas-doctor", "mcp"]
