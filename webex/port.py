"""The real Webex port: how Maya reaches the Webex graph (maya/ports.py describes it).

Both calls run the same graph through run_webex(). They differ only in
wants_filing: a confirmed handoff may file (after an approval), a question never does.
"""

from maya.schemas import PendingAccessResult, WebexHandoff, WebexQuestion
from webex.graph import run_webex
from webex.schemas import AccessDecision, WebexRequest
from webex.tools import Toolbox


def to_result(decision: AccessDecision, idempotency_key: str = "") -> PendingAccessResult:
    """Webex's decision in Maya's terms. The main record is the last thing this run
    wrote (AP005, then AR005), else the first existing one (T001 or AR001)."""
    if decision.actions_taken:
        main_record = decision.actions_taken[-1].record_id
    elif decision.existing_records:
        main_record = decision.existing_records[0]
    else:
        main_record = decision.request_key
    return PendingAccessResult(
        workflow="webex_access",
        status=decision.status,
        request_id=main_record,
        idempotency_key=idempotency_key,
        answer=decision.answer,
        citations=[fact.citation for fact in decision.observed_facts],
    )


class RealWebexPort:
    """Runs Maya's Webex handoffs and questions through the Webex graph."""

    def __init__(self, graph, toolbox: Toolbox):
        self.graph = graph      # from webex.graph.build_graph, with a checkpointer
        self.toolbox = toolbox

    async def request_access(self, handoff: WebexHandoff) -> PendingAccessResult:
        request = WebexRequest(
            caller=handoff.caller, employee_id=handoff.employee_id, software=handoff.software,
            message=f"Please file {handoff.software} access for {handoff.employee_id}. "
                    f"Reason: {handoff.reason}",
            wants_filing=True,
        )
        decision = await run_webex(self.graph, self.toolbox, request)
        return to_result(decision, handoff.idempotency_key)

    async def ask(self, question: WebexQuestion) -> PendingAccessResult:
        request = WebexRequest(
            caller=question.caller, employee_id=question.employee_id,
            software=question.software, message=question.message, wants_filing=False,
        )
        return to_result(await run_webex(self.graph, self.toolbox, request))
