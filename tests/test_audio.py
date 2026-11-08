import numpy as np

from voiceassist.audio import (
    CHUNK_SAMPLES,
    ToneCodec,
    chunk_audio,
    is_speech,
    rms,
    vad_segments,
)

CODEC = ToneCodec()


def test_chunking_fixed_size_and_padding():
    samples = np.random.default_rng(0).normal(size=1000).astype(np.float32)
    chunks = list(chunk_audio(samples))
    assert all(len(c) == CHUNK_SAMPLES for c in chunks)
    assert len(chunks) == 4  # 1000 samples -> 4 chunks of 320


def test_vad_silence_vs_tone():
    assert not is_speech(np.zeros(CHUNK_SAMPLES, dtype=np.float32))
    assert is_speech(CODEC.encode_text("a")[:CHUNK_SAMPLES])


def test_vad_segments_finds_speech_gap():
    silence = [np.zeros(CHUNK_SAMPLES, dtype=np.float32)] * 3
    speech = list(chunk_audio(CODEC.encode_text("hello")))
    chunks = silence + speech + silence
    segments = vad_segments(chunks)
    assert len(segments) == 1
    assert segments[0][0] == 3


def test_codec_round_trip_short():
    text = "hello world"
    assert CODEC.decode_audio(CODEC.encode_text(text)) == text


def test_codec_round_trip_long_with_punctuation_and_digits():
    text = "remember that my meeting is at 3pm, ok?"
    assert CODEC.decode_audio(CODEC.encode_text(text)) == text


def test_codec_normalizes_uppercase():
    assert CODEC.decode_audio(CODEC.encode_text("HELLO")) == "hello"


def test_rms_scales_with_amplitude():
    assert rms(np.ones(100, dtype=np.float32)) == 1.0
    assert rms(np.zeros(100, dtype=np.float32)) == 0.0
