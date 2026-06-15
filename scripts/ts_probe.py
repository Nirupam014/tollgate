#!/usr/bin/env python3
"""Probe: print tree-sitter syntax trees for representative Go/Java/Ruby agents.

This is a one-shot diagnostic to discover the *actual* grammar node types before
writing the per-language recovery parsers (Route B). It is not part of the package.

Run:
    pip install "tree-sitter>=0.21" "tree-sitter-language-pack>=0.4"
    PYTHONPATH=src python scripts/ts_probe.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tollgate.parsers import treesitter_backend as tsb  # noqa: E402

SAMPLES = {
    ("java", "imperative unbounded loop"): '''
class Agent {
  void run(OpenAIClient client, String task) {
    while (true) {
      var r = client.chat().completions().create(params);
      task = r.choices().get(0).message().content();
    }
  }
}
''',
    ("java", "bounded loop + max tokens"): '''
class Agent {
  void run(OpenAIClient client) {
    for (int i = 0; i < 10; i++) {
      var r = client.chat().completions().create(
          ChatCompletionCreateParams.builder().maxCompletionTokens(500).build());
    }
  }
}
''',
    ("java", "langgraph4j builder"): '''
var workflow = new StateGraph<>(schema)
    .addNode("agent", node1)
    .addNode("tools", node2)
    .addEdge(START, "agent")
    .addConditionalEdges("agent", cond, Map.of("tools", "tools", "end", END))
    .addEdge("tools", "agent");
''',
    ("go", "imperative unbounded loop"): '''
package main
func run(client *openai.Client, task string) {
    for {
        resp, _ := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
            Model: openai.GPT4o, Messages: msgs})
        task = resp.Choices[0].Message.Content
    }
}
''',
    ("go", "bounded for + break"): '''
package main
func run(client *openai.Client) {
    for i := 0; i < 10; i++ {
        resp, _ := client.CreateChatCompletion(ctx, req)
        if done { break }
    }
}
''',
    ("ruby", "imperative loop do"): '''
def run(client, task)
  msgs = []
  loop do
    resp = client.chat(parameters: { model: "gpt-4o", messages: msgs })
    msgs << resp.dig("choices", 0, "message")
  end
end
''',
    ("ruby", "times bounded"): '''
def run(client)
  3.times do
    client.chat(parameters: { model: "gpt-4o", max_tokens: 500, messages: msgs })
  end
end
''',
}


def main() -> int:
    if not tsb.available():
        print("tree-sitter not installed. Run:\n"
              '  pip install "tree-sitter>=0.21" "tree-sitter-language-pack>=0.4"',
              file=sys.stderr)
        return 2
    import tree_sitter
    print(f"tree_sitter version: {getattr(tree_sitter, '__version__', '?')}")
    try:
        import tree_sitter_language_pack as tslp
        print(f"tree_sitter_language_pack version: {getattr(tslp, '__version__', '?')}")
    except Exception as e:
        print(f"language pack import note: {e}")
    for (lang, label), src in SAMPLES.items():
        print("\n" + "=" * 78)
        print(f"### {lang.upper()} — {label}")
        print("=" * 78)
        tree = tsb.parse(src, lang)
        print(tsb.sexp(tree.root_node))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
