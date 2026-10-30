"""Interfaces to outside systems: the Webex approval workflow.

A "port" is the one method the graph calls to reach an outside system. The graph
depends on the port, not on the system, so the real workflow can be plugged in
later and tests can pass a fake.
"""

from typing import Protocol

from maya.schemas import PendingAccessResult, WebexHandoff


class WebexPort(Protocol):
    """Anything that can pass a Webex access request to the approval workflow."""

    async def request_access(self, handoff: WebexHandoff) -> PendingAccessResult:
        ...


class FakeWebexPort:
    """A stand-in for the workflow: records every handoff and answers "pending".

    It doesn't deduplicate (the graph must do that), so a test can count exactly how
    many handoffs were sent.
    """

    def __init__(self):
        self.calls: list[WebexHandoff] = []

    async def request_access(self, handoff: WebexHandoff) -> PendingAccessResult:
        self.calls.append(handoff)
        return PendingAccessResult(
            workflow="webex_access",
            status="pending",
            # The same key always gives the same reference, so runs are repeatable.
            request_id="WX-" + handoff.idempotency_key[:8],
            idempotency_key=handoff.idempotency_key,
        )
