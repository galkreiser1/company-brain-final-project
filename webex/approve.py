"""Decide an approval, then resume the Webex run that waits for it.

Run from the project root (the MCP server must be running):
  python -m webex.approve AP005 --as E018 --decision approved
"""

import argparse
import asyncio
import sys

from langgraph.types import Command

from maya.retrieval import OpenSearchRetriever
from webex.graph import build_graph, open_checkpointer, run_config
from webex.tools import ToolError, load_mcp_toolbox


async def approve(approval_id: str, approver_id: str, decision: str, reason: str) -> None:
    toolbox = await load_mcp_toolbox()
    # The server checks that approver_id is the person named on the row, and that
    # the row is still open. If not, this raises and nothing is resumed.
    try:
        [row] = await toolbox.call(
            "record_approval_decision", approval_id=approval_id,
            approver_id=approver_id, decision=decision, reason=reason,
        )
    except ToolError as error:
        sys.exit(f"Not recorded: {error}")
    print(f"{approval_id} is now {row['status']}.")

    # A new approval's request_id is the request key the waiting run is saved under.
    config = run_config(row["request_id"])
    async with open_checkpointer() as saver:
        graph = build_graph(toolbox, OpenSearchRetriever(), checkpointer=saver)
        if not (await graph.aget_state(config)).interrupts:
            print(f"No Webex run is waiting on {approval_id}; nothing to resume.")
            return
        # The resume value is only a note: the wait step re-reads the row above.
        final_state = await graph.ainvoke(Command(resume=decision), config)

    result = final_state["decision"]
    print(f"Run {result.request_key}: {result.status} ({result.outcome})")
    print(result.answer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("approval_id", help="e.g. AP005")
    parser.add_argument("--as", dest="approver_id", required=True, help="your employee id")
    parser.add_argument("--decision", required=True, choices=["approved", "rejected"])
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    asyncio.run(approve(args.approval_id, args.approver_id, args.decision, args.reason))


if __name__ == "__main__":
    main()
