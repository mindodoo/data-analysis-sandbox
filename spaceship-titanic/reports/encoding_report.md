# Encoding Report — Agent 2

No one-hot encoding or target encoding is performed here.

Instead, Agent 2 focuses on:
- type coercion (boolean-like -> int 0/1)
- parsing cabin into structured columns
- adding missing indicator flags
- keeping categorical fields as strings for downstream agents
