"""POSITIVE: a `for` loop that iterates over a queue it grows inside the body.

Looks bounded (a for-loop) but is not: appending to the same list it iterates
makes it run unboundedly. Expected: a CRITICAL recursive_loop and a BLOCK gate.
"""
import anthropic

client = anthropic.Anthropic()
work = ["seed"]


def step(prompt):
    return client.messages.create(
        model="claude-sonnet-4",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )


def run():
    for item in work:
        out = step(item)
        work.append(str(out))       # grows the iterable being looped over
