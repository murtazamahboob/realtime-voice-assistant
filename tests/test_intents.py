from voiceassist.brain import AssistantBrain
from voiceassist.intents import handle, route


def test_route_intents():
    assert route("hello there").name == "greeting"
    assert route("what time is it?").name == "time"
    assert route("set a timer for 5 minutes").name == "timer"
    assert route("calculate 12 * 8").name == "calculate"
    assert route("tell me a joke").name == "joke"
    assert route("who are you").name == "identity"


def test_calculate():
    assert "96" in handle(route("calculate 12 * 8"), {})
    assert "10" in handle(route("calculate 3 + 7"), {})


def test_timer_stored_in_memory():
    memory = {}
    reply = handle(route("set a timer for 5 minutes"), memory)
    assert "5 minutes" in reply
    assert memory["timers"] == ["5 minutes"]


def test_remember_and_recall():
    memory = {}
    handle(route("remember that my meeting is at 3pm"), memory)
    assert "3pm" in handle(route("what do you remember?"), memory)


def test_recall_empty():
    assert "not asked" in handle(route("what do you remember?"), {})


def test_brain_multi_turn_memory():
    brain = AssistantBrain()
    brain.respond("remember that my flight is friday")
    assert "friday" in brain.respond("what do you remember?")
    assert len(brain.history) == 2


def test_brain_empty_transcript_asks_again():
    assert "did not catch" in AssistantBrain().respond("   ")


def test_brain_greeting_uses_name():
    assert "murtaza" in AssistantBrain(user_name="murtaza").respond("hello")
