"""POSITIVE (recommendation): one expensive-model call on a simple task.

No loop, so no gate block — but a frontier model on a short classification
prompt should yield a cheaper-model recommendation. Expected: discovered, PASS
gate, at least one model-substitution recommendation.
"""
import openai


def classify(text):
    # gpt-4-turbo on a trivial label task; the substitution engine should flag a
    # cheaper model that still clears the capability bar.
    return openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": f"Reply with one word, the topic of: {text}"}],
        max_tokens=16,   # capped: this case tests model right-sizing, not output caps
    ).choices[0].message.content
