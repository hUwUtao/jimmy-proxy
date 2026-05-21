#!/usr/bin/env python3
"""Minimal direct chat to chatjimmy.ai — no proxy, no tools, no system prompt."""

import json, urllib.request, ssl, re, sys

URL = "https://chatjimmy.ai/api/chat"
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://chatjimmy.ai",
    "Referer": "https://chatjimmy.ai/",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/148.0.0.0 Safari/537.36",
    "sec-ch-ua": '"Not/A)Brand";v="99", "Chromium";v="148"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Linux"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "priority": "u=1, i",
}
CTX = ssl.create_default_context()

messages = []
print("Chat with Jimmy (Ctrl+D to exit)")
print()

while True:
    try:
        text = input(">>> ")
    except EOFError:
        print()
        break
    if not text:
        continue

    messages.append({"role": "user", "content": text})
    payload = {
        "messages": messages,
        "chatOptions": {"selectedModel": "llama3.1-8B", "systemPrompt": "", "topK": 8},
        "attachment": None,
    }
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers=HEADERS)
    try:
        resp = urllib.request.urlopen(req, timeout=60, context=CTX)
        body = resp.read().decode("utf-8")
    except Exception as e:
        print(f"  ERROR: {e}")
        messages.pop()
        continue

    reply = re.sub(r"<\|stats\|>.*?<\|/stats\|>", "", body, flags=re.DOTALL).strip()
    stats = re.search(r"<\|stats\|>(.*?)<\|/stats\|>", body, re.DOTALL)
    if stats:
        s = json.loads(stats.group(1))
        print(f'  [{s.get("prefill_tokens","?")}→{s.get("decode_tokens","?")} tok] {reply}')
    else:
        print(f"  {reply}")

    messages.append({"role": "assistant", "content": reply})
