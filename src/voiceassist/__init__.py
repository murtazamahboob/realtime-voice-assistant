"""Realtime Voice Assistant — offline speech-to-text, brain, text-to-speech
with barge-in and a latency budget. No microphone, network or API key needed.
"""
from .pipeline import FIRST_AUDIO_BUDGET_MS, TurnResult, VoiceAssistant

__all__ = ["VoiceAssistant", "TurnResult", "FIRST_AUDIO_BUDGET_MS"]
__version__ = "1.0.0"
