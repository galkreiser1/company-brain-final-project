"""Smoke test: run the Maya agent on the first turns of golden session S2.

Needs Lesson 10's MCP server running. Run from the repo root:
    .venv/bin/python -m scripts.smoke_s2

All turns run on one thread_id, so they form one conversation. For each turn it
prints which tools the model called, its answer, and how many messages are saved.
"""

import asyncio
import json

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from maya.config import MCP_SERVER_URL, REPO_ROOT
from maya.graph import build_graph
from maya.planning import message_text


GOLDEN_DATASETS = REPO_ROOT / "data" / "GOLDEN-DATASETS.json"
SESSION_ID = "S2-onboarding-maya"
TURNS_TO_RUN = 3

# The conversation's ID. The checkpointer saves the state under it, so every turn sent
# with the same thread_id continues the same conversation.
THREAD_ID = "smoke-s2"

# LangGraph counts every step it runs in a turn and stops the turn with an error past
# this limit. It stops a model that keeps asking for tools from looping forever.
RECURSION_LIMIT = 12


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


def print_turn(messages: list) -> None:
    """Print the tools the model called in the latest turn, then its final answer."""

    # The latest turn starts at the last user message.
    turn_start = max(
        index for index, message in enumerate(messages)
        if isinstance(message, HumanMessage)
    )
    for message in messages[turn_start:]:
        for call in getattr(message, "tool_calls", None) or []:
            print(f"  tool:   {call['name']}({call['args']})")
    print(f"  answer: {message_text(messages[-1])}")
    print(f"  ({len(messages)} messages saved in this thread)")


async def main() -> None:
    tools = await load_tools()
    print(f"{len(tools)} tools from {MCP_SERVER_URL}:")
    for tool in tools:
        print(f"  - {tool.name}")

    graph = build_graph(tools)

    config = {
        "configurable": {"thread_id": THREAD_ID},
        "recursion_limit": RECURSION_LIMIT,
    }

    for number, user_message in enumerate(load_user_messages(), start=1):
        print(f"\nTurn {number}: {user_message}")
        # Only the new message is sent. The checkpointer adds the earlier ones.
        result = await graph.ainvoke(
            {"messages": [HumanMessage(user_message)]}, config=config,
        )
        print_turn(result["messages"])


if __name__ == "__main__":
    asyncio.run(main())
