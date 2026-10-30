"""The Maya agent's graph: its state, its steps in run order, and the wiring.

Each turn runs these steps:
  1. check  - is it time to distil? (enough new messages since the last time)
  2. distil - (only when step 1 says yes) turn the new messages into updated memory.
  3. plan      - the planner reads the memory + newer messages, and writes the plan;
                 code adds the subject's id and the collections to search.
  4. authorize - may the caller ask about this subject? If not, go to "deny".
  4x. deny     - (when step 4 says no) a fixed refusal, and the turn ends. No model,
                 tool or search runs.
  5. retrieve  - plain code searches the plan's collections with its search_query, as
                 the caller (who is set by code, never by a model).
  6. select    - plain code picks the turn's loadout (the tools the model may see).
  7. model     - the answer call: the plan + the last two turns, with only the
                 loadout's tools attached.
  8. route     - after the model replies, decide what happens next (9a, 9b, or 10).
  9a. tools    - (when the model asked for tools) run them, then back to step 7.
  9b. rearm    - (when needed records weren't looked up) delete that answer, and retry
                 step 7 once with every read tool.
  10. respond  - turn the answer into a MayaReply (answer, citations, maybe a
                 checklist), then code drops every citation it can't back up.

Read in this order: AgentState, then the steps, then build_graph at the bottom.
"""

from functools import partial
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from maya.config import DATA_DIR, RESPOND_MODEL_ID
from maya.ingest import load_employee_ids
from maya.loadout import maya_tools, select_tools
from maya.memory import (
    ConversationMemory,
    distiller_messages,
    memory_is_safe,
    needs_distillation,
    planner_messages,
)
from maya.model import get_model
from maya.planning import (
    ContextPlan,
    TurnPlan,
    answer_messages,
    build_context_plan,
    fallback_plan,
    respond_messages,
    message_text,
    recent_turns,
)
from maya.respond import add_sources, enforce_evidence, tool_result_ids
from maya.retrieval import EvidenceRetriever, authorize_turn
from maya.schemas import CallerContext, Evidence, MayaReply


# STATE -----------------------------------------------------------------------

class AgentState(TypedDict):
    """Everything saved for one conversation.

    `messages` appends: its reducer (add_messages, the function that merges a step's
    update into the old value) adds each new message to the end of the list. The other
    fields are replaced by each update.
    """

    messages: Annotated[list, add_messages]  # the whole conversation
    plan: dict                              # this turn's plan (a ContextPlan, as a dict)
    loadout: list[str]                      # tools the model may see this turn
    rearmed: bool                           # has this turn's fallback been used?
    memory: dict                            # the notes (a ConversationMemory, as a dict)
    distilled_upto: int                     # messages before this index are in the memory
    evidence: list[dict]                    # this turn's document chunks (Evidence, as dicts)
    reply: dict                             # this turn's MayaReply, as a dict
    checklist: dict                         # the latest OnboardingChecklist; kept across turns
    sources_seen: list[str]                 # every chunk_id and record ID seen so far


# THE STEPS -------------------------------------------------------------------
# Each step gets the state, plus whatever build_graph() passes in, and returns only
# the state fields it changes.

def distil_or_plan(state: AgentState) -> str:
    """Step 1: at the start of a turn, decide whether to distil first.

    Returns "distil" if enough new messages have piled up (see needs_distillation),
    otherwise "plan". Most turns skip distillation, so it costs nothing on them.
    """

    if needs_distillation(state):
        return "distil"
    return "plan"


async def distil(state: AgentState, distiller) -> dict:
    """Step 2 (only when step 1 says so): turn the new messages into updated memory.

    - Sends the old memory plus every message since it to the distiller.
    - Keeps the new memory only if memory_is_safe() accepts it. Otherwise nothing
      changes: the old memory and every message stay as they were.
    - Never deletes a message. It only moves `distilled_upto`, so the planner stops
      reading the messages that are now in the memory.
    """

    result = await distiller.ainvoke(distiller_messages(state))
    new_memory = result["parsed"]
    old_memory = state.get("memory") or {}

    if new_memory is None or not memory_is_safe(old_memory, new_memory):
        return {}

    return {
        "memory": new_memory.model_dump(),
        "distilled_upto": len(state["messages"]),  # everything so far is now in memory
    }


async def plan_turn(state: AgentState, planner, employee_ids: dict[str, str]) -> dict:
    """Step 3: ask the planner for this turn's plan, and save it in the state.

    The planner gets the memory plus only the newer messages (planner_messages in
    memory.py). Code then adds the subject's id and the collections to search
    (build_context_plan). If the planner's answer can't be read as a TurnPlan, use a
    fallback plan: topic "other", documents and records both needed, no subject.
    """

    result = await planner.ainvoke(planner_messages(state))
    turn_plan = result["parsed"]
    if not turn_plan:
        turn_plan = fallback_plan()
    plan = build_context_plan(turn_plan, employee_ids)
    return {"plan": plan.model_dump()}


def caller_from_config(config: RunnableConfig) -> CallerContext | None:
    """Who is asking, from the run's config. None if caller_id or user_group is missing.

    The caller is set by the code that starts the turn (config["configurable"]), never
    read from a message or a model's output, so no prompt can change it.
    """

    settings = config.get("configurable", {})
    caller_id = settings.get("caller_id")
    user_group = settings.get("user_group")
    if not caller_id or not user_group:
        return None
    return CallerContext(caller_id=caller_id, user_group=user_group)


# The deny path's whole answer. Fixed text, so it can't reveal anything about anyone.
REFUSAL = "I can't share another employee's onboarding information."


def authorize_or_deny(state: AgentState, config: RunnableConfig) -> str:
    """Step 4: may this caller ask about the employee the message is about?

    Returns "retrieve" (go on) or "deny" (see authorize_turn in retrieval.py). It runs
    before any search, tool or answer call: the MCP tools don't check who is asking,
    so stopping the whole turn is the only way to leak nothing.
    """

    plan = ContextPlan(**state["plan"])
    if authorize_turn(caller_from_config(config), plan):
        return "retrieve"
    return "deny"


def deny(state: AgentState) -> dict:
    """Step 4x (only when step 4 says no): reply with the fixed refusal. No model call.

    Also clears the turn's evidence and loadout, so nothing from the last turn is left
    looking like this turn's. The reply has no citations and no checklist.
    """

    return {
        "messages": [AIMessage(REFUSAL)],
        "reply": MayaReply(answer=REFUSAL).model_dump(),
        "evidence": [],
        "loadout": [],
        "rearmed": False,
    }


def missing_collections(required: list[str], evidence: list[Evidence]) -> list[str]:
    """The required collections (the plan's required_evidence) the evidence lacks."""

    found = {chunk.collection for chunk in evidence}
    return [name for name in required if name not in found]


def merge_evidence(first: list[Evidence], extra: list[Evidence]) -> list[Evidence]:
    """The first search's chunks, then the extra search's new ones (no duplicates)."""

    seen = {chunk.chunk_id for chunk in first}
    return first + [chunk for chunk in extra if chunk.chunk_id not in seen]


def retrieve(state: AgentState, config: RunnableConfig, retriever: EvidenceRetriever) -> dict:
    """Step 5: search the documents for this turn. No model call of its own.

    - Unknown caller: no evidence ("fail closed": no search beats a search as nobody).
    - The plan requires no collections (no documents needed, or topic "other"): no
      search.
    - Otherwise: one search with the plan's search_query (or the user's message, if
      the planner left it empty).
    - If the results lack a collection the topic needs, one more search, limited to
      the missing collections. Never a third, even if that one finds nothing.
    Always sets `evidence`, so last turn's chunks never carry over.

    LangGraph passes `config` in because the function has a parameter of that name.
    """

    caller = caller_from_config(config)
    plan = ContextPlan(**state["plan"])
    if caller is None or not plan.required_evidence:
        return {"evidence": []}

    query = plan.search_query or message_text(state["messages"][-1])
    evidence = retriever.search(query, caller)

    missing = missing_collections(plan.required_evidence, evidence)
    if missing:
        extra = retriever.search(query, caller, collections=missing)
        evidence = merge_evidence(evidence, extra)
    return {"evidence": [chunk.model_dump() for chunk in evidence]}


def select(state: AgentState) -> dict:
    """Step 6: turn the plan into this turn's loadout (select_tools). No model call.

    Also resets `rearmed` for the new turn.
    """

    return {"loadout": select_tools(ContextPlan(**state["plan"])), "rearmed": False}


def model_with_tools(base_model, tools_by_name: dict, names: list[str]):
    """Return the model with exactly these tools attached (no names: no tools)."""

    if not names:
        return base_model
    chosen_tools = [tools_by_name[name] for name in sorted(names)]
    return base_model.bind_tools(chosen_tools)


def call_model(state: AgentState, base_model, tools_by_name: dict) -> dict:
    """Step 7: ask the model for an answer, with only the loadout's tools.

    It gets the plan plus the last two turns (answer_messages in planning.py).
    Returns the model's reply, which may be an answer or a request to call tools.
    """

    model = model_with_tools(base_model, tools_by_name, state["loadout"])
    plan = ContextPlan(**state["plan"])  # the state stores the plan as a plain dict
    response = model.invoke(answer_messages(state, plan))
    return {"messages": [response]}


def used_tool_this_turn(messages: list) -> bool:
    """True if any tool ran in the current turn (since the latest user message)."""

    current_turn = recent_turns(messages, keep=1)
    return any(isinstance(message, ToolMessage) for message in current_turn)


def lookup_was_missed(state: AgentState) -> bool:
    """True if the plan needed records, no tool ran, and we haven't retried yet.

    Only records count: documents come from the retrieve step, not from tools.
    """

    plan = ContextPlan(**state["plan"])
    return (
        plan.needs_records
        and not state["rearmed"]
        and not used_tool_this_turn(state["messages"])
    )


def route(state: AgentState) -> str:
    """Step 8: after the model replies, choose the next step.

    - The model asked for tools          -> "tools" (step 9a)
    - It answered, but missed a lookup   -> "rearm" (step 9b, retry once)
    - It answered                        -> "respond" (step 10, then the turn is over)
    """

    if state["messages"][-1].tool_calls:
        return "tools"
    if lookup_was_missed(state):
        return "rearm"
    return "respond"


def rearm(state: AgentState, all_tool_names: list[str]) -> dict:
    """Step 9b (only when a lookup was missed): retry once with every read tool.

    - Loadout: all of Maya's tools (only read tools are ever loaded).
    - Deletes the answer the model just wrote without a lookup (the user never saw it).
    - Sets `rearmed`, so route() won't send this turn here a second time.
    """

    unseen_answer = state["messages"][-1]
    return {
        "loadout": all_tool_names,
        "rearmed": True,
        "messages": [RemoveMessage(id=unseen_answer.id)],  # delete by id, don't append
    }


def reply_args(raw_message) -> dict:
    """The responder's reply before any checks: the arguments of its MayaReply call.

    Structured output works through a tool call named after the schema, so the reply
    is that call's arguments. Raises an error if the model made no such call.
    """

    return raw_message.tool_calls[0]["args"]


def respond(state: AgentState, responder) -> dict:
    """Step 10: turn the answer into a MayaReply, keeping only citations code can check.

    1. code:  add this turn's chunk_ids and record IDs to sources_seen.
    2. model: the responder fills a MayaReply (respond_messages in planning.py).
    3. code:  enforce_evidence drops citations not in sources_seen, and the checklist
              if this topic doesn't want one. It works on the raw reply, so one bad
              item can't sink the whole reply.
    If the reply can't be read as a MayaReply (or the call fails), the turn still
    ends: with the model's answer, no citations and no checklist.
    """

    plan = ContextPlan(**state["plan"])
    current_turn = recent_turns(state["messages"], keep=1)
    chunk_ids = [chunk["chunk_id"] for chunk in state.get("evidence") or []]
    sources_seen = add_sources(state.get("sources_seen") or [],
                               chunk_ids + tool_result_ids(current_turn))
    # Code decides which turns get a checklist, not the model.
    checklist_wanted = plan.current_intent == "onboarding_status"

    try:
        result = responder.invoke(respond_messages(state, plan, checklist_wanted))
        # The answer the user reads is the answer model's, not a rewrite of it.
        raw = {**reply_args(result["raw"]), "answer": message_text(state["messages"][-1])}
        reply = enforce_evidence(raw, sources_seen, checklist_wanted, plan.subject_employee_id)
    except Exception:  # no reply, or one that still breaks a rule: never fail the turn
        reply = MayaReply(answer=message_text(state["messages"][-1]))

    update = {"reply": reply.model_dump(), "sources_seen": sources_seen}
    if reply.checklist is not None:
        update["checklist"] = reply.checklist.model_dump()  # replaces the saved one
    return update


# WIRING ----------------------------------------------------------------------

# LangGraph counts every step it runs in a turn and stops the turn with an error past
# this limit. It stops a model that keeps asking for tools from looping forever.
RECURSION_LIMIT = 12


def build_graph(tools: list, retriever: EvidenceRetriever):
    """Create the models, then connect the steps into a graph.

    `tools` can be everything the MCP server offers: only Maya's five read tools are
    kept (maya_tools), so no step can ever see another one.
    `retriever` searches the documents (an OpenSearchRetriever, or a fake in tests).

    `partial(f, x=...)` makes a copy of f with x already filled in, so LangGraph can
    call each step with just the state.
    """

    tools = maya_tools(tools)  # first, so everything below sees only these
    base_model = get_model()  # tools are attached per turn, in call_model
    tools_by_name = {tool.name: tool for tool in tools}
    all_tool_names = [tool.name for tool in tools]
    # The planner must reply with a TurnPlan. include_raw=True returns a dict whose
    # "parsed" is None when the reply can't be read as one, instead of raising an error.
    planner = get_model().with_structured_output(TurnPlan, include_raw=True)
    # {full name: employee id}, read once here, to fill each plan's subject_employee_id.
    employee_ids = load_employee_ids(DATA_DIR)
    distiller = get_model().with_structured_output(ConversationMemory, include_raw=True)
    # A stronger model for this one call: see RESPOND_MODEL_ID in config.py.
    responder = get_model(RESPOND_MODEL_ID).with_structured_output(MayaReply, include_raw=True)

    graph = StateGraph(AgentState)
    graph.add_node("distil", partial(distil, distiller=distiller))
    graph.add_node("plan", partial(plan_turn, planner=planner, employee_ids=employee_ids))
    graph.add_node("deny", deny)
    graph.add_node("retrieve", partial(retrieve, retriever=retriever))
    graph.add_node("select", select)
    graph.add_node("model", partial(
        call_model, base_model=base_model, tools_by_name=tools_by_name,
    ))
    graph.add_node("rearm", partial(rearm, all_tool_names=all_tool_names))
    graph.add_node("respond", partial(respond, responder=responder))
    # Step 9a: LangGraph's ToolNode runs the tools the model asked for.
    graph.add_node("tools", ToolNode(tools))

    # START -> (distil) -> plan -> (authorize) -> retrieve -> select -> model, then
    # model -> tools / rearm / respond -> END, and tools and rearm go back to model.
    # A denied turn goes plan -> deny -> END instead.
    graph.add_conditional_edges(
        START, distil_or_plan, {"distil": "distil", "plan": "plan"},
    )
    graph.add_edge("distil", "plan")
    graph.add_conditional_edges(
        "plan", authorize_or_deny, {"retrieve": "retrieve", "deny": "deny"},
    )
    graph.add_edge("deny", END)
    graph.add_edge("retrieve", "select")
    graph.add_edge("select", "model")
    graph.add_conditional_edges(
        "model", route, {"tools": "tools", "rearm": "rearm", "respond": "respond"},
    )
    graph.add_edge("respond", END)
    graph.add_edge("tools", "model")
    graph.add_edge("rearm", "model")

    # The checkpointer saves the state per conversation (thread_id) between turns, so
    # each turn starts with the messages of all earlier turns.
    # with_config sets the limit on every run, so callers don't have to remember it.
    compiled = graph.compile(checkpointer=InMemorySaver())
    return compiled.with_config(recursion_limit=RECURSION_LIMIT)
