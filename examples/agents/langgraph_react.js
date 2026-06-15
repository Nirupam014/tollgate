// LangGraph.js ReAct agent — an agent<->tools cycle with no iteration bound.
// Tollgate recovers this StateGraph into the same IR as a Python LangGraph and
// flags the unbounded cycle as a critical, uncapped-cost finding.
import { StateGraph, START, END, MessagesAnnotation } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({ model: "gpt-4o" });

function shouldContinue(state) {
  const last = state.messages[state.messages.length - 1];
  return last.tool_calls?.length ? "tools" : "end";
}

async function callModel(state) {
  const response = await model.invoke(state.messages);
  return { messages: [...state.messages, response] };
}

const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", shouldContinue, { tools: "tools", end: END })
  .addEdge("tools", "agent");   // back-edge -> cycle, no termination guard

export const app = workflow.compile();
