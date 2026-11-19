"""The retry after a missed lookup (rearm) must call a tool. (No model calls.)

    .venv/bin/python -m pytest -v tests/test_maya_rearm.py

When the plan needs records and the model answers without calling a tool, rearm
deletes that answer and the model tries again. On that retry the model is bound with
tool_choice "any", so it can't answer from documents or an earlier message alone.
Once a tool has run, it may answer.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from maya.graph import model_with_tools, must_call_tool


def state(rearmed: bool, *after_question) -> dict:
    return {"rearmed": rearmed,
            "messages": [HumanMessage("How do I get my licenses sorted?"), *after_question]}


TOOL_RESULT = ToolMessage("{'status': 'blocked'}", tool_call_id="1", name="check_software_subscription")


def test_the_first_try_may_answer_without_a_tool():
    assert must_call_tool(state(rearmed=False)) is False


def test_the_retry_must_call_a_tool():
    assert must_call_tool(state(rearmed=True)) is True


def test_after_the_tool_ran_the_retry_may_answer():
    # Otherwise the model could only ever call tools, and the turn would never end.
    called = AIMessage("", tool_calls=[{"name": "check_software_subscription", "args": {}, "id": "1"}])
    assert must_call_tool(state(True, called, TOOL_RESULT)) is False


def test_an_earlier_turns_tool_does_not_count():
    # A tool from the previous turn is history: this turn still has to look it up.
    messages = [HumanMessage("earlier"), TOOL_RESULT, HumanMessage("How do I get my licenses sorted?")]
    assert must_call_tool({"rearmed": True, "messages": messages}) is True


class FakeModel:
    """Records how bind_tools was called."""

    def bind_tools(self, tools, **options):
        self.bound = (tools, options)
        return self


def test_must_call_tool_binds_tool_choice_any():
    model = FakeModel()
    model_with_tools(model, {"t": "tool-t"}, ["t"], must_call_tool=True)
    assert model.bound == (["tool-t"], {"tool_choice": "any"})


def test_a_normal_call_binds_no_tool_choice():
    model = FakeModel()
    model_with_tools(model, {"t": "tool-t"}, ["t"])
    assert model.bound == (["tool-t"], {})
