// Go agent — two LLM calls per turn (plan → act) inside an unbounded loop.
// Recovers as a 2-node chain (plan → act) with a back-edge cycle.
package main

import "github.com/sashabaranov/go-openai"

func run(client *openai.Client) {
	for { // unbounded
		plan, _ := client.CreateChatCompletion(ctx, planReq)
		act, _ := client.CreateChatCompletion(ctx, actReq)
		_ = plan
		_ = act
	}
}
