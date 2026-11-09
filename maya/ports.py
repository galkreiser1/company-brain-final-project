"""Interfaces to outside systems: the Webex workflow (webex/).

A "port" is the one method the graph calls to reach an outside system. The graph
depends on the port, not on the system, so the real workflow can be plugged in
later and tests can pass a fake.
"""

from typing import Protocol

from maya.schemas import PendingAccessResult, WebexHandoff, WebexQuestion


class WebexPort(Protocol):
    """Anything that can pass Webex work on: the real workflow (webex/port.py) or a fake."""

    async def request_access(self, handoff: WebexHandoff) -> PendingAccessResult:
        """File access (a confirmed go-ahead)."""
        ...

    async def ask(self, question: WebexQuestion) -> PendingAccessResult:
        """Answer a Webex question. Files nothing."""
        ...


class FakeWebexPort:
    """A stand-in for the workflow: records every handoff and answers "pending".

    It doesn't deduplicate (the graph must do that), so a test can count exactly how
    many handoffs were sent.
    """

    def __init__(self):
        self.calls: list[WebexHandoff] = []
        self.questions: list[WebexQuestion] = []

    async def request_access(self, handoff: WebexHandoff) -> PendingAccessResult:
        self.calls.append(handoff)
        return PendingAccessResult(
            workflow="webex_access",
            status="pending",
            # The same key always gives the same reference, so runs are repeatable.
            request_id="WX-" + handoff.idempotency_key[:8],
            idempotency_key=handoff.idempotency_key,
        )

    async def ask(self, question: WebexQuestion) -> PendingAccessResult:
        self.questions.append(question)
        return PendingAccessResult(
            workflow="webex_access", status="completed", request_id="WX-question",
            answer=f"(fake Webex answer about {question.employee_id})",
        )
