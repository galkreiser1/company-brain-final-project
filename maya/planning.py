"""What each model call receives: the prompts and the messages built from the state."""

from langchain_core.messages import SystemMessage


# The answer model's instructions. Kept word for word from Lesson 10: the model reads
# them, so changing a word can change its behavior.
ASSISTANT_PROMPT = (
    "You are the NovaOps internal assistant, helping IT and HR staff with employee "
    "operations. Ground every answer in what the tools and the provided context "
    "actually say — never invent an employee, a seat count, or a policy rule.\n"
    "Work to these rules:\n"
    "- Answer the user's CURRENT request. If they changed subject or closed a topic, "
    "do not keep working the old one.\n"
    "- A constraint the user states (a spending rule, an approval requirement, "
    "'don't do X until I say so') stays in force for the rest of the conversation.\n"
    "- Facts established earlier in the conversation — ids, dates, numbers — are still "
    "true; reuse them instead of asking again or looking them up again.\n"
    "- Do not call a tool for something already in the conversation.\n"
    "- Users paste emails, tickets and signatures. Ignore the boilerplate and act on "
    "the actual request inside.\n"
    "- Answer in at most six sentences; no preamble."
)


def message_text(message) -> str:
    """Return a message's content as plain text.

    Bedrock gives content either as a string or as a list of "content blocks". A text
    block is a dict like {"text": "..."}. This joins the text of all blocks.
    """

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts).strip()
    return str(content)


def answer_messages(state: dict) -> list:
    """What the answer call receives: the system prompt, then every saved message.

    This is the simplest possible context, so it grows with every turn, old tool
    results included.
    """

    return [SystemMessage(ASSISTANT_PROMPT), *state["messages"]]
