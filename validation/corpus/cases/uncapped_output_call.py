"""POSITIVE (strict lint): a single, bounded LLM call with NO output-token cap.

Nothing here loops or fans out — the only risk is that the response is uncapped,
so one call can run to the model's full context window. Under the strict agentic
linter this is an `uncapped_output` finding and a WARN gate (not a block: it is a
soft, per-call cost risk, not an unbounded series). This fixture pins that
behavior so a future change can't silently stop flagging it.
"""
import openai

client = openai.OpenAI()


def summarize(text):
    # No max_tokens / max_output_tokens -> uncapped generation.
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Summarize:\n{text}"}],
    ).choices[0].message.content
