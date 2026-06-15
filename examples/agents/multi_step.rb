# Ruby agent — two LLM calls per turn (plan -> act) inside an unbounded loop.
# Recovers as a 2-node chain with a cycle.
require "openai"

def run(client)
  msgs = []
  loop do
    plan = client.chat(parameters: { model: "gpt-4o", messages: msgs })
    act  = client.chat(parameters: { model: "gpt-4o", messages: msgs })
  end
end
