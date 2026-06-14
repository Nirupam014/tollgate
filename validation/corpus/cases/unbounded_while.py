"""POSITIVE: a hand-rolled agent with an unbounded `while True` driver loop.

This is the canonical unbounded-cost shape. The loop has no break and grows a
task queue every turn, so token spend is uncapped. Expected: a CRITICAL
recursive_loop and a BLOCK gate.
"""
import openai

OBJECTIVE = "keep working forever"
queue = ["start"]


def llm(prompt, model="gpt-4o"):
    return openai.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content


def main():
    while True:                      # no break -> unbounded
        task = queue.pop(0)
        result = llm(f"{OBJECTIVE}: {task}")
        for line in str(result).splitlines():
            queue.append(line)       # queue grows every iteration


if __name__ == "__main__":
    main()
