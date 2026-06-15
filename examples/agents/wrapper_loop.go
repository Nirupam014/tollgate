// Go agent — the loop calls a thin wrapper (callLLM) that issues the SDK call.
// Recovery resolves the wrapper chain: the node is sited at `callLLM`, and the
// SDK call inside the wrapper is not double-counted.
package main

import "github.com/sashabaranov/go-openai"

func callLLM(client *openai.Client, task string) string {
	resp, _ := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: openai.GPT4o, Messages: msgs})
	return resp.Choices[0].Message.Content
}

func run(client *openai.Client, task string) {
	for { // unbounded loop around the wrapper
		task = callLLM(client, task)
	}
}
