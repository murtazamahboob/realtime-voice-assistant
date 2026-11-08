"""Intent routing for the assistant: what the user wants, from their words.

Rule-based on purpose — the router must be instant, explainable and
testable offline. Each intent is a small matcher plus a handler that gets
the assistant's memory, so multi-turn behaviour (notes, timers) lives here
instead of in a black box.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Intent:
    name: str
    slots: dict = field(default_factory=dict)


def _safe_eval(expr: str):
    """Evaluate a plain arithmetic expression; nothing else is allowed."""
    tree = ast.parse(expr, mode="eval")
    allowed = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd, ast.FloorDiv,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed) or (
            isinstance(node, ast.Constant) and not isinstance(node.value, (int, float))
        ):
            raise ValueError("not a plain arithmetic expression")
    return eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})


def route(text: str) -> Intent:
    t = text.strip().lower()
    if re.search(r"\b(hello|hi|hey|good (morning|afternoon|evening))\b", t):
        return Intent("greeting")
    if "what time" in t or "time is it" in t:
        return Intent("time")
    if m := re.search(r"timer (?:for )?(\d+)\s*(second|minute|hour)", t):
        return Intent("timer", {"amount": int(m.group(1)), "unit": m.group(2)})
    if m := re.search(r"(?:remember|note|remind me) (?:that )?(.+)", t):
        return Intent("remember", {"note": m.group(1).strip()})
    if re.search(r"what (did i (ask|say)|is my note|are my notes|do you remember)", t):
        return Intent("recall")
    if m := re.search(r"(?:calculate|what is|whats)\s+([\d\s+\-*/().%]+)$", t):
        return Intent("calculate", {"expr": m.group(1).strip()})
    if "joke" in t:
        return Intent("joke")
    if "weather" in t:
        return Intent("weather")
    if re.search(r"(your name|who are you)", t):
        return Intent("identity")
    if re.search(r"\b(stop|goodbye|bye|exit|quit)\b", t):
        return Intent("goodbye")
    return Intent("chat", {"text": t})


def handle(intent: Intent, memory: dict) -> str:
    """Produce the reply for a routed intent, updating ``memory`` as needed."""
    notes = memory.setdefault("notes", [])
    if intent.name == "greeting":
        name = memory.get("user_name")
        return f"hello{' ' + name if name else ''}! how can i help?"
    if intent.name == "time":
        return f"it is {datetime.now().strftime('%I:%M %p').lstrip('0')}."
    if intent.name == "timer":
        amount, unit = intent.slots["amount"], intent.slots["unit"]
        memory.setdefault("timers", []).append(f"{amount} {unit}{'s' if amount != 1 else ''}")
        return f"timer set for {amount} {unit}{'s' if amount != 1 else ''}."
    if intent.name == "remember":
        notes.append(intent.slots["note"])
        return f"noted: {intent.slots['note']}."
    if intent.name == "recall":
        if not notes:
            return "you have not asked me to remember anything yet."
        return "you asked me to remember: " + "; ".join(notes) + "."
    if intent.name == "calculate":
        try:
            value = _safe_eval(intent.slots["expr"])
            return f"{intent.slots['expr']} is {value}."
        except Exception:
            return "i could not calculate that, try a plain sum like 12 plus 8."
    if intent.name == "joke":
        return "why do programmers prefer dark mode? because light attracts bugs."
    if intent.name == "weather":
        return "i am offline, so i have no live weather. check outside or connect a weather source."
    if intent.name == "identity":
        return "i am your realtime voice assistant, running fully offline."
    if intent.name == "goodbye":
        return "goodbye! talk soon."
    text = intent.slots.get("text", "")
    return f"you said: {text}. i am listening, ask me anything."
