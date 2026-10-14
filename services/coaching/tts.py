import os
from io import BytesIO

from groq import Groq


class TextToSpeech:
    """Groq Orpheus TTS (WAV) with automatic gTTS fallback (MP3)."""

    MODEL = "canopylabs/orpheus-v1-english"
    VOICE = "autumn"

    def __init__(self, client=None):
        self.client = client

        if self.client is None:
            api_key = os.environ.get("GROQ_API_KEY", "")
            if api_key:
                self.client = Groq(api_key=api_key)

    def _speak_groq(self, text):
        response = self.client.audio.speech.create(
            model=self.MODEL,
            voice=self.VOICE,
            input=text,
            response_format="wav",
        )
        return response.read()

    def _speak_gtts(self, text, lang="en"):
        from gtts import gTTS

        buffer = BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(buffer)
        buffer.seek(0)
        return buffer.read()

    def speak(self, text, lang="en"):
        """Return audio bytes (WAV from Groq, or MP3 from gTTS fallback)."""
        cleaned = (text or "").strip()
        if not cleaned:
            return None

        if self.client is not None:
            try:
                return self._speak_groq(cleaned)
            except Exception as e:
                print(f"[tts] Groq TTS failed, falling back to gTTS: {type(e).__name__}: {e} | cause: {getattr(e, '__cause__', None)}")

        try:
            return self._speak_gtts(cleaned, lang)
        except Exception as e:
            print(f"[tts] gTTS failed: {e}")
            return None