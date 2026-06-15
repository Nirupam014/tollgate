// Go agent — an unbounded `for {}` loop around an LLM call with no output cap.
// Go isn't parsed into a graph (that needs the tree-sitter backend), so Tollgate's
// language-agnostic textual lint flags it: unbounded_loop + uncapped_output.
package main

import "github.com/sashabaranov/go-openai"

func run(client *openai.Client, task string) {
	for { // no break / no max-iteration bound
		resp, _ := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
			Model:    openai.GPT4o,
			Messages: msgs, // no MaxTokens set -> uncapped generation
		})
		task = resp.Choices[0].Message.Content
	}
}
