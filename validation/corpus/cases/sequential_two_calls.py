"""HARD NEGATIVE: a straight-line script with two LLM calls and no loop.

Expected: two LLM nodes, NO recursive_loop finding, PASS gate. Guards against a
parser that fabricates a cycle wherever it sees repeated calls.
"""
import openai


def ask(prompt, model="gpt-4o-mini"):
    return openai.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], max_tokens=256
    ).choices[0].message.content


def main():
    classified = ask("classify this ticket")
    summary = ask(f"summarize: {classified}")
    print(summary)
