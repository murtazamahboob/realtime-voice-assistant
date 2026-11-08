"""CLI: `voiceassist turn "hello"` runs one turn; `voiceassist demo` runs a
scripted conversation and writes the latency ledger + reply audio.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .audio import write_wav
from .pipeline import VoiceAssistant

DEMO_SCRIPT = [
    "hello",
    "remember that my meeting is at 3pm",
    "what do you remember?",
    "calculate 12 * 8",
    "set a timer for 5 minutes",
    "what time is it?",
    "tell me a joke",
]


def cmd_turn(args) -> int:
    assistant = VoiceAssistant()
    result = assistant.run_turn(" ".join(args.words))
    print(f"you: {result.transcript}")
    print(f"assistant: {result.reply}")
    print(f"intent: {result.intent}  first audio: {result.latency_ms['first_audio_ms']} ms  "
          f"total: {result.latency_ms['total_ms']} ms  within budget: {result.within_budget}")
    return 0


def cmd_demo(args) -> int:
    assistant = VoiceAssistant(user_name="murtaza")
    results = assistant.run_conversation(DEMO_SCRIPT)
    for r in results:
        print(f"you: {r.transcript}\nassistant: {r.reply}  "
              f"[{r.intent}, first audio {r.latency_ms['first_audio_ms']} ms]")
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/latency.json", "w") as f:
        json.dump(assistant.ledger, f, indent=2)
    last_audio = assistant.codec.encode_text(results[-1].reply)
    write_wav("outputs/last_reply.wav", last_audio)
    report = (
        "# Demo report — Realtime Voice Assistant\n\n"
        + "\n".join(
            f"- Turn {i}: you said \"{r.transcript}\" -> \"{r.reply}\" "
            f"({r.intent}, first audio {r.latency_ms['first_audio_ms']} ms, "
            f"total {r.latency_ms['total_ms']} ms)"
            for i, r in enumerate(results, 1)
        )
        + f"\n\n- Turns: {len(results)}, all within the 500 ms first-audio budget: "
        + str(all(r.within_budget for r in results))
        + "\n- Last reply audio: outputs/last_reply.wav\n"
    )
    with open("outputs/demo_report.md", "w") as f:
        f.write(report)
    print(f"demo done: {len(results)} turns, report=outputs/demo_report.md")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="voiceassist")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("turn", help="run one spoken turn")
    p.add_argument("words", nargs="+")
    sub.add_parser("demo", help="run the scripted demo conversation")
    args = parser.parse_args(argv)
    return cmd_demo(args) if args.cmd == "demo" else cmd_turn(args)
