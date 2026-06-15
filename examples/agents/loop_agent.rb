# Ruby agent — an unbounded `loop do` around an LLM call with no output cap.
# Recovered by the language-agnostic textual lint (no graph parse for Ruby yet):
# unbounded_loop + uncapped_output.
require "openai"

def run(client, task)
  msgs = [{ role: "user", content: task }]
  loop do                                # no break / no max-iteration bound
    resp = client.chat(parameters: { model: "gpt-4o", messages: msgs })  # no max_tokens
    msgs << resp.dig("choices", 0, "message")
  end
end
