// Java agent — two LLM calls per turn (plan → act) inside an unbounded loop.
// Recovers as a 2-node chain with a cycle.
import com.openai.client.OpenAIClient;

class MultiStep {
  void run(OpenAIClient client) {
    while (true) {
      var plan = client.chat().completions().create(planParams);
      var act = client.chat().completions().create(actParams);
    }
  }
}
