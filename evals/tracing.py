"""Langfuse labels for the runners: one labelled trace per message.

The graphs trace themselves (the callback in maya/graph.py records every step). What
they can't know is the run around them: who is asking, which case and turn this is,
and how the turn ended. The runners add that here, so a reviewer can find a turn in
Langfuse by case and turn, and read its status first.
"""

from contextlib import contextmanager
from datetime import datetime

from langfuse import get_client, propagate_attributes

import maya.config  # noqa: F401  (loads the LANGFUSE_* keys from .env first)

# Names this run, in the record file and in Langfuse, so its traces can be found.
RUN_STAMP = f"{datetime.now():%Y%m%d-%H%M%S}"

langfuse = get_client()


def require_langfuse() -> None:
    """Stop before any model call if Langfuse rejects the keys: it would record nothing,
    silently."""
    if not langfuse.auth_check():
        raise SystemExit("Langfuse rejected the keys in .env, so this run would record nothing.")


@contextmanager
def traced_turn(name: str, message: str, *, conversation: str, user_id: str, tags: list[str],
                metadata: dict):
    """One trace for one message. Use as `with traced_turn(...) as root:`.

    - session_id: groups a conversation's turns into one Langfuse session; the run
      stamp keeps each run's sessions apart.
    - user_id, tags and metadata: filters, e.g. "every trace for E010" or "case W-I-03".
      propagate_attributes puts them on every step recorded inside the `with`.
    - the root span: the trace's top step, with the message as input. "Current" means
      the graphs' steps, recorded inside it, nest under it.
    """
    with propagate_attributes(
        session_id=f"{conversation}-{RUN_STAMP}",
        user_id=user_id,
        tags=tags,
        metadata={**metadata, "run": RUN_STAMP},
    ):
        with langfuse.start_as_current_observation(as_type="agent", name=name, input=message) as root:
            yield root


def finish_turn(root, answer: str, metadata: dict) -> str | None:
    """Put the answer and the outcome on the root span; return the trace's link."""
    root.update(output=answer, metadata=metadata)
    return langfuse.get_trace_url()


def turn_status(refused: bool, webex_results: list) -> str:
    """The turn's outcome in one word: Webex's own status if Webex ran, else Maya's."""
    if webex_results:
        return webex_results[-1].status
    return "refused" if refused else "answered"
