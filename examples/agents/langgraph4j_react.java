// LangGraph4j ReAct agent (Java) — an agent<->tools cycle with no bound.
// With the optional tree-sitter backend, Tollgate recovers this StateGraph into
// the same IR as a Python/JS LangGraph and flags the unbounded cycle.
import org.bsc.langgraph4j.StateGraph;

class ReactAgent {
  StateGraph<State> build() {
    return new StateGraph<>(schema)
        .addNode("agent", this::callModel)
        .addNode("tools", this::callTools)
        .addEdge(START, "agent")
        .addConditionalEdges("agent", this::shouldContinue,
            Map.of("tools", "tools", "end", END))
        .addEdge("tools", "agent");   // back-edge -> cycle, no termination guard
  }
}
