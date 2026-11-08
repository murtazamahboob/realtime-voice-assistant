"""Streaming text-to-speech over the tone codec.

``OfflineTTS`` turns reply text into audio and yields it in 20 ms chunks,
exactly as a streaming TTS service would. Synthesis is interruptible:
calling :meth:`OfflineTTS.stop` (or setting the shared ``stop_event``)
halts the stream at the next chunk boundary — that is what makes barge-in
possible. ``first_chunk_latency_ms`` is measured by the caller.
"""
from __future__ import annotations

import threading

import numpy as np

from .audio import CHUNK_SAMPLES, ToneCodec, chunk_audio


class OfflineTTS:
    """Chunk-streaming, interruptible TTS backed by :class:`ToneCodec`."""

    def __init__(self, codec=None):
        self.codec = codec or ToneCodec()
        self.stop_event = threading.Event()
        self.chunks_spoken = 0
        self.stopped_early = False

    def stop(self) -> None:
        self.stop_event.set()

    def reset(self) -> None:
        self.stop_event.clear()
        self.chunks_spoken = 0
        self.stopped_early = False

    def synthesize(self, text: str):
        """Yield 20 ms audio chunks for ``text`` until done or stopped."""
        self.reset()
        audio = self.codec.encode_text(text)
        for chunk in chunk_audio(audio):
            if self.stop_event.is_set():
                self.stopped_early = True
                break
            self.chunks_spoken += 1
            yield chunk

    def synthesize_all(self, text: str) -> np.ndarray:
        """Render the whole reply (or up to the stop point) as one array."""
        chunks = list(self.synthesize(text))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)
