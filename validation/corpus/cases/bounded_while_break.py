"""HARD NEGATIVE: a `while True` that *looks* scary but has a real break.

A grep-for-`while True` baseline flags this; Tollgate must not call it CRITICAL,
because the break bounds it. Expected: a loop finding at MEDIUM severity at most
(verify-the-bound), and NOT a BLOCK gate.
"""
import openai


def llm(prompt):
    return openai.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
    )


def main():
    steps = 0
    while True:
        llm("take one more step")
        steps += 1
        if steps >= 5:
            break                    # bounded: terminates after 5 iterations
