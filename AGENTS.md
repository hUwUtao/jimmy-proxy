# Jimmy Proxy

Single-file Python bridge (`proxy.py`) translating OpenAI-compatible API to chatjimmy.ai.

## Run

```bash
python proxy.py                          # default :4100
python proxy.py --log                    # with file logging to proxy.log
python proxy.py --port 4100 --log --log-file custom.log
```

No deps, no build, no venv required — stdlib only (HTTP server, not ASGI/WSGI).

## Test

```bash
# List models
curl http://localhost:4100/v1/models

# Chat completion
curl http://localhost:4100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama3.1-8B","messages":[{"role":"user","content":"hi"}]}'

# Streaming
curl http://localhost:4100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama3.1-8B","messages":[{"role":"user","content":"hi"}],"stream":true}'

# Multi-turn diagnostic (bypasses proxy, hits upstream directly)
echo -e "My name is Bob\nWhat is my name?" | python3 chat.py

# OpenResponses API (used by OpenClaw)
curl http://localhost:4100/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama3.1-8B","input":[{"type":"message","role":"user","content":"hi"}]}'
```

## Quirks & constraints

- **System prompt limit ~24K chars** — upstream returns empty past this. Proxy auto-replaces bloated prompts (e.g. from OpenCode) with a minimal one so tool definitions aren't garbled by abrupt truncation.
- **Supports both** `/v1/chat/completions` and `/v1/responses` (OpenClaw)
- **Tool calling** via `func("args")` flat format (no nested JSON, weak-model friendly)
- **No streaming from upstream** — full response buffered before returning to client; TTFT = full generation time
- **Tools filtered** before forwarding: `webfetch`, `todowrite`, `skill`, `question`, `task` are stripped
- **No auth** — upstream is open beta, no API key
- **Only model**: `llama3.1-8B` (3-bit/6-bit quantized)
- Binds to `127.0.0.1` only
- **Threaded server** — uses `ThreadingHTTPServer`; concurrent requests don't block each other
- **No formatting standard** — project uses no formatter, linter, or type checker

## opencode.jsonc

`opencode.jsonc` in this repo is a **template** for other projects — copy it to their root, not used by this repo itself.
