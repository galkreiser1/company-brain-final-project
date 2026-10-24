"""Smoke test: run the Maya agent on the first turns of golden session S2.

Needs Lesson 10's MCP server running. Run from the repo root:
    .venv/bin/python -m scripts.smoke_s2

For now it only connects to the server, lists its tools, and shows the turns it
will send. The agent itself comes in the next step.
"""

import asyncio
import json

from langchain_mcp_adapters.client import MultiServerMCPClient

from maya.config import MCP_SERVER_URL, REPO_ROOT


GOLDEN_DATASETS = REPO_ROOT / "data" / "GOLDEN-DATASETS.json"
SESSION_ID = "S2-onboarding-maya"
TURNS_TO_RUN = 3


async def load_tools() -> list:
    """Connect to the MCP server and get its NovaOps tools as LangChain tools."""

    client = MultiServerMCPClient(
        {"novaops": {"url": MCP_SERVER_URL, "transport": "streamable_http"}}
    )
    return await client.get_tools()


def load_user_messages() -> list[str]:
    """The user's messages for the first TURNS_TO_RUN turns of session S2."""

    sessions = json.loads(GOLDEN_DATASETS.read_text())["sessions"]
    session = next(s for s in sessions if s["id"] == SESSION_ID)
    return [turn["user"] for turn in session["turns"][:TURNS_TO_RUN]]


async def main() -> None:
    tools = await load_tools()
    print(f"{len(tools)} tools from {MCP_SERVER_URL}:")
    for tool in tools:
        print(f"  - {tool.name}")

    print(f"\nTurns to send ({SESSION_ID}):")
    for number, message in enumerate(load_user_messages(), start=1):
        print(f"  {number}. {message}")


if __name__ == "__main__":
    asyncio.run(main())
