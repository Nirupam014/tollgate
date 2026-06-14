"""HARD NEGATIVE: a classic bounded ReAct-style loop (`for _ in range(N)`).

This is the shape that should NOT be flagged: a fixed iteration count with no
queue growth. Expected: no recursive_loop finding and a PASS gate.
"""
import openai

MAX_STEPS = 6


def llm(prompt):
    return openai.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}], max_tokens=256
    ).choices[0].message.content


def react(question):
    scratch = question
    for _ in range(MAX_STEPS):       # fixed, bounded; iterable is not grown
        scratch = llm(f"Thought/Action over: {scratch}")
        if "FINISH" in str(scratch):
            break
    return scratch
