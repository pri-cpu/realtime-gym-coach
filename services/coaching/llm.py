from groq import APIConnectionError

from services.config.workout_config import PROMPT


def describe_error(e):
    """Readable error text, including the underlying cause (e.g. SSL/DNS/header)."""
    text = f"{type(e).__name__}: {e}"
    cause = getattr(e, "__cause__", None)
    if cause:
        text += f" | cause: {type(cause).__name__}: {cause}"
    return text


class LLMCoach:
    MODEL = "openai/gpt-oss-120b"

    FALLBACK_TEXT = {
        "workout_started": "Let's go! Get into position and start when you're ready.",
        "set_completed": "Nice set! Take a short rest and get ready for the next one.",
        "workout_completed": "Great workout! Well done today.",
    }

    def __init__(self, groq_client):
        self.client = groq_client
        self.history = []
        self.system_prompt = PROMPT

    def _fallback(self, event, issue):
        if event in self.FALLBACK_TEXT:
            return self.FALLBACK_TEXT[event]
        if issue:
            return "Watch your form and keep your movements controlled."
        return "Keep going, you're doing great!"

    def _chat(self, messages):
        # gpt-oss is a reasoning model: keep reasoning short so replies are fast.
        try:
            return self.client.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                temperature=0.4,
                reasoning_effort="low",
                max_completion_tokens=300,
            )
        except APIConnectionError:
            raise  # network problem: retrying with other settings won't help
        except Exception as e:
            print(f"[llm] fast-mode call failed ({describe_error(e)}); retrying with defaults")
            return self.client.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                temperature=0.4,
            )

    def give_feedback(self, event, issue):
        prompt = f"Event: {event}"

        if issue:
            prompt += f" Form Issue: {issue}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.history[-10:],
            {"role": "user", "content": prompt},
        ]

        text = ""
        try:
            response = self._chat(messages)
            text = (response.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[llm] Groq chat failed: {describe_error(e)}")

        if not text:
            # Never crash or stay silent: use a short preset line instead.
            return self._fallback(event, issue)

        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": text})

        return text