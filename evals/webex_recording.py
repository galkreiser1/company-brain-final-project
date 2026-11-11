"""The real Webex port, wrapped so a runner can see what happened on each turn.

The wrappers have the same methods as what they wrap, so the graphs can't tell the
difference. They record:
  - each handoff Maya sent and each question Maya passed on (RecordingWebexPort);
  - each tool Webex called, with its arguments (RecordingToolbox).
"""

from contextlib import asynccontextmanager

from webex.graph import build_graph, open_checkpointer
from webex.port import RealWebexPort
from webex.tools import McpToolbox


class RecordingToolbox:
    def __init__(self, toolbox):
        self.toolbox = toolbox
        self.calls: list[dict] = []  # {"name": ..., "args": {...}}, in order

    async def call(self, name: str, **args) -> list:
        self.calls.append({"name": name, "args": args})
        return await self.toolbox.call(name, **args)


class RecordingWebexPort:
    def __init__(self, port, toolbox: RecordingToolbox):
        self.port = port
        self.toolbox = toolbox
        self.calls = []      # WebexHandoff: what Maya filed (same name as FakeWebexPort's)
        self.questions = []  # WebexQuestion: what Maya asked
        self.results = []    # PendingAccessResult: what Webex answered, to either

    @property
    def tool_calls(self) -> list[dict]:
        return self.toolbox.calls

    async def request_access(self, handoff):
        self.calls.append(handoff)
        result = await self.port.request_access(handoff)
        self.results.append(result)
        return result

    async def ask(self, question):
        self.questions.append(question)
        result = await self.port.ask(question)
        self.results.append(result)
        return result


@asynccontextmanager
async def real_webex_port(tools: list, retriever):
    """A recorded RealWebexPort over the MCP server's tools. Webex's paused runs are
    saved to runs/webex_checkpoints.sqlite while the `async with` block is open."""
    toolbox = RecordingToolbox(McpToolbox(tools))
    async with open_checkpointer() as saver:
        graph = build_graph(toolbox, retriever, checkpointer=saver)
        yield RecordingWebexPort(RealWebexPort(graph, toolbox), toolbox)
