"""The realtime pipeline: listen -> understand -> speak, with barge-in.

One turn is: encode the user's words as audio (the simulated microphone),
stream them through VAD + streaming STT, route the transcript through the
brain, then stream the reply out through interruptible TTS. Every turn is
timed — STT time, brain time, and time-to-first-audio — and written to a
latency ledger, because a voice assistant that answers slowly is a voice
assistant nobody uses. The ledger's budget is 500 ms to first audio.

Barge-in: while the reply is streaming, new user speech stops the TTS
mid-sentence (``interrupt_after_chunks`` in :meth:`VoiceAssistant.run_turn`
simulates the user starting to talk again) and the new utterance is handled
as the next turn instead of being talked over.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .audio import ToneCodec, chunk_audio
from .brain import AssistantBrain
from .stt import OfflineSTT
from .tts import OfflineTTS

FIRST_AUDIO_BUDGET_MS = 500.0


@dataclass
class TurnResult:
    transcript: str
    reply: str
    intent: str | None = None
    audio_chunks_spoken: int = 0
    interrupted: bool = False
    partials_seen: int = 0
    latency_ms: dict = field(default_factory=dict)
    within_budget: bool = True


class VoiceAssistant:
    """A realtime voice assistant that runs fully offline."""

    def __init__(self, user_name: str | None = None):
        self.codec = ToneCodec()
        self.brain = AssistantBrain(user_name=user_name)
        self.tts = OfflineTTS(codec=self.codec)
        self.ledger: list[dict] = []

    def run_turn(self, user_text: str, interrupt_after_chunks: int | None = None) -> TurnResult:
        """Run one full turn from the user's spoken words.

        ``user_text`` is the scripted utterance (the simulated microphone).
        ``interrupt_after_chunks`` simulates barge-in: the user starts
        speaking again after that many reply chunks, cutting the reply off.
        """
        t0 = time.perf_counter()
        audio = self.codec.encode_text(user_text)
        stt = OfflineSTT(codec=self.codec)
        partials = 0
        for chunk in chunk_audio(audio):
            partials += len(list(stt.feed(chunk)))
        transcript = stt.finalize().text
        t_stt = time.perf_counter()

        reply = self.brain.respond(transcript)
        intent = self.brain.history[-1]["intent"] if self.brain.history else None
        t_brain = time.perf_counter()

        # Stream the reply; stop early on barge-in. Setting stop() makes the
        # generator end at the next chunk boundary with stopped_early set.
        spoken, first_audio_ms, interrupted = 0, None, False
        for chunk in self.tts.synthesize(reply):
            if first_audio_ms is None:
                first_audio_ms = (time.perf_counter() - t_brain) * 1000
            spoken += 1
            if interrupt_after_chunks is not None and spoken >= interrupt_after_chunks:
                self.tts.stop()
                interrupted = True
        t_end = time.perf_counter()

        latency = {
            "stt_ms": round((t_stt - t0) * 1000, 2),
            "brain_ms": round((t_brain - t_stt) * 1000, 2),
            "first_audio_ms": round(first_audio_ms or 0.0, 2),
            "total_ms": round((t_end - t0) * 1000, 2),
        }
        result = TurnResult(
            transcript=transcript,
            reply=reply,
            intent=intent,
            audio_chunks_spoken=spoken,
            interrupted=interrupted or self.tts.stopped_early,
            partials_seen=partials,
            latency_ms=latency,
            within_budget=(first_audio_ms or 0.0) <= FIRST_AUDIO_BUDGET_MS,
        )
        self.ledger.append({"turn": len(self.ledger) + 1, **latency,
                            "intent": intent, "interrupted": result.interrupted})
        return result

    def run_conversation(self, utterances: list[str]) -> list[TurnResult]:
        return [self.run_turn(u) for u in utterances]
