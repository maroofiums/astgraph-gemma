"""Mode 2b: the same tools as an MCP server, so any MCP client (IDE, desktop agent) can use them.

    pip install mcp
    python integrations/mcp_server.py /path/to/repo
"""
import sys
try:                                    # mcp >= 2
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:                     # mcp 1.x
    from mcp.server.fastmcp import FastMCP
from astgraph import RepoIndex
from astgraph.tools import ToolKit


def build(repo: str) -> FastMCP:
    kit = ToolKit(RepoIndex.from_path(repo))
    mcp = FastMCP("astgraph")

    @mcp.tool()
    def repo_map(query: str, budget: int = 1024) -> str:
        """Ranked map (file paths + signatures) of the repository for a problem description. Call first."""
        return kit.repo_map(query, budget)

    @mcp.tool()
    def find_symbol(name: str) -> str:
        """Look up a class/method/function by 'name', 'Class.method' or 'path.py::name'."""
        return kit.find_symbol(name)

    @mcp.tool()
    def read_symbol(name: str, max_lines: int = 60) -> str:
        """Source of one symbol."""
        return kit.read_symbol(name, max_lines)

    @mcp.tool()
    def callers(name: str) -> str:
        """Symbols that call the given symbol (name-based, approximate)."""
        return kit.callers(name)

    @mcp.tool()
    def callees(name: str) -> str:
        """Symbols called by the given symbol (name-based, approximate)."""
        return kit.callees(name)

    @mcp.tool()
    def outline(path: str) -> str:
        """All symbols defined in one file with line ranges."""
        return kit.outline(path)

    return mcp


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".").run()
