"""How the Webex agent calls the MCP server's tools.

The steps call `toolbox.call(name, **args)` and get plain Python values back, so
they don't deal with MCP's message format. Tests pass a toolbox that calls the
server's functions directly instead (no server running).
"""

from typing import Protocol

from maya.respond import json_values

# The only tools Webex may call: an allowlist, as in maya/loadout.py. The write
# tools are added in later steps, together with the code that guards them.
WEBEX_READ_TOOLS = (
    "get_employee",
    "check_software_subscription",
    "list_employee_tickets",
    "list_access_requests",
    "list_approvals",
)


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
        self.tools = {tool.name: tool for tool in tools if tool.name in WEBEX_READ_TOOLS}

    async def call(self, name: str, **args) -> list:
        if name not in self.tools:
            raise ValueError(f"Webex may not call {name!r}.")
        content = await self.tools[name].ainvoke(args)
        # A tool that returns a list sends each item as its own JSON block.
        return json_values(content_text(content))
