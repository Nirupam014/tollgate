// Hand-rolled JS agent — an unbounded `while (true)` loop around an LLM call,
// with no output-token cap. Tollgate recovers the self-loop (critical recursive
// loop) and merges the uncapped-output lint finding.
import OpenAI from "openai";

const openai = new OpenAI();

export async function run(task) {
  let messages = [{ role: "user", content: task }];
  while (true) {                       // no break / max-iteration bound
    const r = await openai.chat.completions.create({
      model: "gpt-4o",
      messages,                        // history grows every turn
    });
    messages.push(r.choices[0].message);
    task = r.choices[0].message.content;
  }
}
