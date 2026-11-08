"""Streaming speech-to-text over the tone codec.

``OfflineSTT`` consumes 20 ms audio chunks exactly as a microphone stream
would deliver them, runs VAD, and emits partial transcripts as each
character-frame completes — so callers see words appear while the user is
still speaking, not only at the end. ``finalize`` returns the full text.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import CHAR_SAMPLES, CHUNK_SAMPLES, chunk_audio, is_speech


@dataclass
class PartialTranscript:
    text: str
    is_final: bool
    frames_decoded: int


class OfflineSTT:
    """Chunk-streaming STT backed by :class:`~voiceassist.audio.ToneCodec`."""

    def __init__(self, codec=None):
        from .audio import ToneCodec

        self.codec = codec or ToneCodec()
        self._buffer = np.zeros(0, dtype=np.float32)
        self._text = ""
        self._frames = 0
        self._heard_speech = False

    def feed(self, chunk: np.ndarray):
        """Feed one 20 ms chunk; yields a partial transcript per decoded frame."""
        chunk = np.asarray(chunk, dtype=np.float32)
        if is_speech(chunk):
            self._heard_speech = True
        self._buffer = np.concatenate([self._buffer, chunk])
        while len(self._buffer) >= CHAR_SAMPLES:
            frame, self._buffer = self._buffer[:CHAR_SAMPLES], self._buffer[CHAR_SAMPLES:]
            ch = self.codec.decode_frame(frame)
            self._frames += 1
            # Trailing spaces are only meaningful mid-utterance; strip on finalize.
            self._text += ch
            yield PartialTranscript(text=self._text.strip(), is_final=False, frames_decoded=self._frames)

    def finalize(self) -> PartialTranscript:
        return PartialTranscript(
            text=self._text.strip(), is_final=True, frames_decoded=self._frames
        )

    @property
    def heard_speech(self) -> bool:
        return self._heard_speech


def transcribe_stream(samples: np.ndarray, codec=None) -> list[PartialTranscript]:
    """Convenience helper: stream whole audio through the STT, return partials."""
    stt = OfflineSTT(codec=codec)
    partials = []
    for chunk in chunk_audio(np.asarray(samples, dtype=np.float32)):
        partials.extend(stt.feed(chunk))
    partials.append(stt.finalize())
    return partials
