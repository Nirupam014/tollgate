// Java agent — an unbounded `while (true)` loop around an LLM call with no
// output cap. Recovered by the language-agnostic textual lint (no graph parse for
// Java yet): unbounded_loop + uncapped_output.
import com.openai.client.OpenAIClient;

class LoopAgent {
  void run(OpenAIClient client, String task) {
    while (true) {                       // no break / no max-iteration bound
      var r = client.chat().completions().create(params);   // no maxCompletionTokens
      task = r.choices().get(0).message().content();
    }
  }
}
