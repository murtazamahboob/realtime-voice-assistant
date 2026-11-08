from voiceassist.audio import ToneCodec, chunk_audio
from voiceassist.stt import OfflineSTT, transcribe_stream
from voiceassist.tts import OfflineTTS

CODEC = ToneCodec()


def test_stt_streams_partials_that_grow():
    partials = transcribe_stream(CODEC.encode_text("hello world"))
    finals = [p for p in partials if p.is_final]
    assert len(finals) == 1 and finals[0].text == "hello world"
    non_final = [p.text for p in partials if not p.is_final]
    assert non_final and non_final[-1] == "hello world"
    assert all(len(a) <= len(b) for a, b in zip(non_final, non_final[1:]))


def test_stt_heard_speech_flag():
    stt = OfflineSTT()
    for chunk in chunk_audio(CODEC.encode_text("hi")):
        list(stt.feed(chunk))
    assert stt.heard_speech
    assert stt.finalize().text == "hi"


def test_stt_empty_audio_is_empty():
    import numpy as np

    partials = transcribe_stream(np.zeros(3200, dtype="float32"))
    assert partials[-1].text == ""


def test_tts_renders_and_decodes_back():
    tts = OfflineTTS()
    audio = tts.synthesize_all("hello there")
    assert len(audio) > 0
    assert CODEC.decode_audio(audio) == "hello there"


def test_tts_stop_cuts_synthesis_short():
    from voiceassist.audio import chunk_audio

    text = "this is a very long reply that keeps going"
    full_chunks = len(list(chunk_audio(ToneCodec().encode_text(text))))
    tts = OfflineTTS()
    gen = tts.synthesize(text)
    chunks = [next(gen), next(gen)]
    tts.stop()
    rest = list(gen)
    assert tts.stopped_early
    assert len(chunks) + len(rest) < full_chunks
