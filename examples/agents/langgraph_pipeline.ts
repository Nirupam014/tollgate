// LangGraph.js linear pipeline (TypeScript) — a bounded, acyclic DAG.
// No cycle, so Tollgate passes it: extract -> summarize -> END.
import { StateGraph, START, END } from "@langchain/langgraph";

interface State {
  text: string;
  summary: string;
}

const graph = new StateGraph<State>({ channels: { text: null, summary: null } })
  .addNode("extract", extractFn)
  .addNode("summarize", summarizeFn)
  .addEdge(START, "extract")
  .addEdge("extract", "summarize")
  .addEdge("summarize", END);

export const pipeline = graph.compile();
