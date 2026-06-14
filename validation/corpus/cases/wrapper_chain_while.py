"""POSITIVE: BabyAGI-shaped agent where the loop calls LLM work through wrappers.

The driver `while True` in `main` never touches the SDK directly — it calls
`execution_agent` / `planner`, which call `complete`, which calls the SDK. The
analyzer must recover the LLM calls transitively. Expected: CRITICAL recursive
loop, BLOCK gate, and three distinct LLM nodes in the cycle.
"""
import openai

tasks = ["plan the work"]


def complete(prompt, model="gpt-4o"):
    return openai.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content


def execution_agent(task):
    return complete(f"Do this task: {task}", model="gpt-4o")


def planner(last):
    return complete(f"Given {last}, list next tasks", model="gpt-4o")


def prioritizer(last):
    return complete(f"Reprioritize given {last}", model="gpt-4o-mini")


def main():
    while True:
        t = tasks.pop(0)
        result = execution_agent(t)
        for nt in str(planner(result)).splitlines():
            tasks.append(nt)
        prioritizer(result)


if __name__ == "__main__":
    main()
