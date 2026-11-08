"""How the Webex agent calls the MCP server's tools.

The steps call `toolbox.call(name, **args)` and get plain Python values back, so
they don't deal with MCP's message format. Tests pass a toolbox that calls the
server's functions directly instead (no server running).
"""

from typing import Protocol

from langchain_mcp_adapters.client import MultiServerMCPClient

from maya.config import MCP_SERVER_URL
from maya.respond import json_values

# The only tools Webex may call: an allowlist, as in maya/loadout.py.
WEBEX_READ_TOOLS = (
    "get_employee",
    "check_software_subscription",
    "list_employee_tickets",
    "list_access_requests",
    "list_approvals",
)
# Writes to approvals only. The graph asks (request_approval); the approve command
# records a person's decision. Filing (create_access_request) comes with the write gate.
WEBEX_WRITE_TOOLS = ("request_approval", "record_approval_decision")
WEBEX_TOOLS = WEBEX_READ_TOOLS + WEBEX_WRITE_TOOLS


class ToolError(Exception):
    """A tool refused or failed, e.g. "AP005 must be decided by E018, not E006."."""


class Toolbox(Protocol):
    async def call(self, name: str, **args) -> list:
        """Run one tool. Returns its result as a list of JSON values."""
        ...


def content_text(content) -> str:
    """A LangChain tool result as text: a string, or a list of text blocks."""
    if isinstance(content, str):
        return content
    return "".join(block["text"] for block in content if block.get("type") == "text")


class McpToolbox:
    """The real toolbox: the MCP server's tools, loaded as LangChain tools."""

    def __init__(self, tools: list):
        self.tools = {tool.name: tool for tool in tools if tool.name in WEBEX_TOOLS}

    async def call(self, name: str, **args) -> list:
        if name not in self.tools:
            raise ValueError(f"Webex may not call {name!r}.")
        text = content_text(await self.tools[name].ainvoke(args))
        # A tool that returns a list sends each item as its own JSON block.
        values = json_values(text)
        # The MCP library returns a tool's error as plain text, not as an exception.
        # Raise it, so a refusal can never look like an empty result.
        if not values and text.strip():
            raise ToolError(text)
        return values


async def load_mcp_toolbox() -> McpToolbox:
    """Connect to the MCP server (MCP_SERVER_URL) and keep Webex's tools."""
    client = MultiServerMCPClient({"novaops": {"url": MCP_SERVER_URL, "transport": "streamable_http"}})
    return McpToolbox(await client.get_tools())
