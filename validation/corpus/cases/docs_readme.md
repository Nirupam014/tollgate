# Example Project

This is project documentation, not an agent. It even mentions
`openai.chat.completions.create` in a code sample, and a `while True` loop in
prose, but it is a Markdown doc.

A scan must NOT surface this as a workflow. Expected: NOT discovered.

```python
# illustrative only
while True:
    openai.chat.completions.create(model="gpt-4o", messages=[])
```
