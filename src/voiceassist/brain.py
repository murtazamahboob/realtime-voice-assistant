"""The conversation brain: memory + routing, no network, no API key.

``AssistantBrain`` is deliberately small: it keeps the conversation history
and notes in memory, routes each transcript through :mod:`intents`, and
returns the reply. Swapping in a real LLM means replacing one method —
:py:meth:`AssistantBrain.respond` — not rewiring the pipeline.
"""
from __future__ import annotations

from .intents import handle, route


class AssistantBrain:
    """Stateful, offline brain: route the transcript, remember, reply."""

    def __init__(self, user_name: str | None = None):
        self.memory: dict = {"notes": [], "timers": []}
        if user_name:
            self.memory["user_name"] = user_name
        self.history: list[dict] = []

    def respond(self, transcript: str) -> str:
        transcript = transcript.strip()
        if not transcript:
            return "i did not catch that, could you say it again?"
        intent = route(transcript)
        reply = handle(intent, self.memory)
        self.history.append({"user": transcript, "assistant": reply, "intent": intent.name})
        return reply

    def set_user_name(self, name: str) -> None:
        self.memory["user_name"] = name
