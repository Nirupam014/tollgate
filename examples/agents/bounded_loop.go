// Go agent — a *bounded* condition loop (`for turns < max`) around an LLM call.
// Recovers as a guarded cycle: the loop can terminate, so the recursive-loop is
// flagged as guarded (verify the bound), not as critical/unbounded.
package main

import "github.com/sashabaranov/go-openai"

func run(client *openai.Client) {
	turns := 0
	for turns < 10 { // bounded by the counter condition
		client.CreateChatCompletion(ctx, req)
		turns++
	}
}
