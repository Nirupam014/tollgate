"""Example LangGraph-style agent the analyzer can statically parse.

This file is illustrative input for the parser; it does not need its imports to
resolve for the static AST analysis to recover the graph.
"""
from langgraph.graph import StateGraph, END  # noqa: F401


def build():
    g = StateGraph(dict)
    g.add_node("plan", plan_node)
    g.add_node("act", act_node)
    g.add_node("reflect", reflect_node)
    g.set_entry_point("plan")
    g.add_edge("plan", "act")
    g.add_edge("act", "reflect")
    # Back-edge with no guard: the analyzer flags this as an unbounded loop.
    g.add_conditional_edges("reflect", route, {"continue": "plan", "done": END})
    return g.compile()


def plan_node(state):
    # Appends to messages each turn -> history growth signal.
    state["messages"].append(call_model("claude-opus-4", state["messages"]))
    return state


def act_node(state):
    return state


def reflect_node(state):
    state["messages"].append(call_model("claude-opus-4", state["messages"]))
    return state


def route(state):
    return "continue"


def call_model(model, messages):
    return {"role": "assistant", "content": "..."}
