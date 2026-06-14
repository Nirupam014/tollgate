"""Imperative (hand-rolled) agent in the BabyAGI style.

Not a framework graph — a plain ``while True`` loop that drives three LLM-backed
agents and grows its own task queue every turn. This is the shape most real
agents take, and it is exactly the unbounded-cost risk Tollgate flags: the loop
has no termination guard, so token spend is uncapped under adverse inputs.
"""
import time

import openai

OBJECTIVE = "Plan and execute a research project"
task_list = [{"task_name": "Draft an initial plan"}]


def openai_call(prompt, model="gpt-4o"):
    resp = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def execution_agent(objective, task):
    prompt = f"Objective: {objective}\nTask: {task}\nDo it and report the result."
    return openai_call(prompt, model="gpt-4o")


def task_creation_agent(objective, last_result):
    prompt = f"Objective: {objective}\nLast result: {last_result}\nList new tasks."
    return openai_call(prompt, model="gpt-4o")


def prioritization_agent(objective):
    prompt = f"Objective: {objective}\nReprioritize the task list."
    return openai_call(prompt, model="gpt-4o-mini")


def main():
    while True:                                   # no break -> unbounded loop
        task = task_list.pop(0)
        result = execution_agent(OBJECTIVE, task["task_name"])
        new_tasks = task_creation_agent(OBJECTIVE, result)
        for nt in str(new_tasks).splitlines():
            task_list.append({"task_name": nt})    # queue grows every iteration
        prioritization_agent(OBJECTIVE)
        time.sleep(1)


if __name__ == "__main__":
    main()
