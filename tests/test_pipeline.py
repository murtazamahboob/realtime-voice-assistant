from voiceassist.pipeline import FIRST_AUDIO_BUDGET_MS, VoiceAssistant


def test_full_turn_greeting():
    result = VoiceAssistant().run_turn("hello")
    assert result.transcript == "hello"
    assert "hello" in result.reply
    assert result.intent == "greeting"
    assert result.partials_seen > 0
    assert result.audio_chunks_spoken > 0
    assert not result.interrupted


def test_first_audio_within_budget():
    result = VoiceAssistant().run_turn("hello")
    assert result.latency_ms["first_audio_ms"] <= FIRST_AUDIO_BUDGET_MS
    assert result.within_budget


def test_multi_turn_memory_across_turns():
    assistant = VoiceAssistant()
    assistant.run_turn("remember that my meeting is at 3pm")
    result = assistant.run_turn("what do you remember?")
    assert "3pm" in result.reply


def test_barge_in_cuts_reply_short():
    assistant = VoiceAssistant()
    full = VoiceAssistant().run_turn("tell me a joke")
    cut = assistant.run_turn("tell me a joke", interrupt_after_chunks=2)
    assert cut.interrupted
    assert cut.audio_chunks_spoken == 2
    assert cut.audio_chunks_spoken < full.audio_chunks_spoken


def test_barge_in_then_new_turn_works():
    assistant = VoiceAssistant()
    assistant.run_turn("tell me a joke", interrupt_after_chunks=2)
    result = assistant.run_turn("calculate 12 * 8")
    assert "96" in result.reply
    assert not result.interrupted


def test_conversation_ledger_records_every_turn():
    assistant = VoiceAssistant()
    assistant.run_conversation(["hello", "calculate 3 + 7", "goodbye"])
    assert len(assistant.ledger) == 3
    assert all(entry["first_audio_ms"] <= FIRST_AUDIO_BUDGET_MS for entry in assistant.ledger)


def test_deterministic_same_input_same_output():
    a = VoiceAssistant().run_turn("calculate 12 * 8")
    b = VoiceAssistant().run_turn("calculate 12 * 8")
    assert a.transcript == b.transcript and a.reply == b.reply
