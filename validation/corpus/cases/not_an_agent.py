"""HARD NEGATIVE: ordinary Python with no LLM activity at all.

Even though it has a `while True`, there is no SDK call, so it is not an agent
workflow. Expected: NOT discovered by a scan (no false positive on plain code).
"""
import time


def serve():
    while True:                      # a server loop, not an agent loop
        do_work()
        time.sleep(1)


def do_work():
    total = 0
    for i in range(100):
        total += i * i
    return total
