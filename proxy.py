#!/usr/bin/env python3
"""
Proxy server that translates OpenAI-compatible API requests
to chatjimmy.ai's custom format and back.

Usage:
    python proxy.py [--port 4100] [--log] [--log-file proxy.log]

Then point OpenCode at http://localhost:4100/v1
Logs are written to proxy.log (full request/response details).
"""

import json
import time
import uuid
import argparse
import logging
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import urllib.request
import ssl
import re
import ast


UPSTREAM_URL = "https://chatjimmy.ai/api/chat"
DEFAULT_MODEL = "llama3.1-8B"
FILTERED_TOOLS = {"webfetch", "todowrite", "skill", "question", "task"}
MODELS = {
    "llama3.1-8B": "llama3.1-8B",
}


def _first_sentence(text):
    """Return the first sentence (or first 120 chars) of a description."""
    if not text:
        return ""
    # Cut at first period followed by space/newline, or first newline
    for end in (". ", ".\n", "\n"):
        idx = text.find(end)
        if idx != -1:
            return text[: idx + 1].strip()
    return text[:120].strip()


def format_tools_for_prompt(tools, tool_choice=None):
    """Convert OpenAI tool definitions into a Llama-friendly system-prompt section."""
    if not tools:
        return ""

    lines = [
        "",
        "# Tools",
        "You can run tools by writing a simple Python-style function call.",
        "To use a tool, write its name followed by arguments in parentheses.",
        "You can specify keyword arguments (e.g. `write_file(AbsolutePath='test.txt', Content='Hello')`) or single positional arguments (e.g. `run_command('ls -l')`).",
        "",
        "Rules:",
        "1. Write the tool call on a new line.",
        "2. Do NOT write XML tags or JSON blocks for tools.",
        "3. Only use the available tools listed below.",
        "",
        "Available Tools:",
    ]

    if tool_choice == "none":
        lines.append("Do NOT use tools for this request.")
        lines.append("")
    elif tool_choice == "required":
        lines.append("You MUST call at least one tool.")
        lines.append("")
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        fname = tool_choice.get("function", {}).get("name", "")
        if fname:
            lines.append(f"You MUST call '{fname}'.")
            lines.append("")

    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function")
        if not func:
            continue
        name = func.get("name", "")
        desc = _first_sentence(func.get("description", ""))
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = set(params.get("required", []))
        parts = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "string")
            opt = "" if pname in required else "?"
            parts.append(f"{pname}{opt}: {ptype}")
        sig = ", ".join(parts)
        line = f"- {name}({sig})"
        if desc:
            line += f" — {desc}"
        lines.append(line)

    lines.append("")
    return "\n".join(lines)



def _tool_schema_index(tools):
    index = {}
    for tool in tools or []:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        name = func.get("name")
        if name:
            index[name] = func.get("parameters", {}) or {}
    return index


def _default_for_type(ptype):
    if ptype == "string":
        return ""
    if ptype == "integer":
        return 0
    if ptype == "number":
        return 0
    if ptype == "boolean":
        return False
    if ptype == "array":
        return []
    if ptype == "object":
        return {}
    return ""


def _normalize_tool_args(name, raw_args, schema):
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError:
            raw_args = {}
    if not isinstance(raw_args, dict):
        raw_args = {}

    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = schema.get("required", []) if isinstance(schema, dict) else []

    for key in required:
        if key not in raw_args or raw_args[key] is None:
            pinfo = props.get(key, {})
            raw_args[key] = _default_for_type(pinfo.get("type", "string"))
        else:
            pinfo = props.get(key, {})
            ptype = pinfo.get("type", "string")
            if ptype == "string" and not isinstance(raw_args[key], str):
                raw_args[key] = str(raw_args[key])
            elif ptype == "integer" and not isinstance(raw_args[key], int):
                try:
                    raw_args[key] = int(raw_args[key])
                except (ValueError, TypeError):
                    raw_args[key] = 0
            elif ptype == "number" and not isinstance(raw_args[key], (int, float)):
                try:
                    raw_args[key] = float(raw_args[key])
                except (ValueError, TypeError):
                    raw_args[key] = 0
            elif ptype == "boolean" and not isinstance(raw_args[key], bool):
                raw_args[key] = bool(raw_args[key])
            elif ptype == "array" and not isinstance(raw_args[key], list):
                raw_args[key] = [raw_args[key]]
            elif ptype == "object" and not isinstance(raw_args[key], dict):
                raw_args[key] = {}

    return raw_args


def find_function_calls(text, valid_tools=None):
    """
    Finds all function calls of the form: name(args) in the text.
    Handles quotes and nested parentheses properly.
    Returns list of tuples (name, args_str, start_pos, end_pos).
    """
    calls = []
    n = len(text)
    i = 0
    while i < n:
        # Match an identifier
        m = re.match(r"\b(\w+)\s*\(", text[i:])
        if not m:
            i += 1
            continue
        
        name = m.group(1)
        start_idx = i
        # The arguments start after the '('
        arg_start = i + m.end()
        
        if valid_tools is not None and name not in valid_tools:
            i = arg_start
            continue
            
        # Now scan forward to find the matching ')'
        j = arg_start
        nest_level = 1
        in_single_quote = False
        in_double_quote = False
        escaped = False
        
        while j < n:
            char = text[j]
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif not in_single_quote and not in_double_quote:
                if char == '(':
                    nest_level += 1
                elif char == ')':
                    nest_level -= 1
                    if nest_level == 0:
                        # Found matching parenthesis!
                        args_str = text[arg_start:j]
                        calls.append((name, args_str, start_idx, j + 1))
                        break
            j += 1
        
        # Advance i
        if j < n:
            i = j + 1
        else:
            i += 1
            
    return calls


def parse_with_ast(args_str):
    try:
        # Wrap in a dummy function call to parse it as an expression
        tree = ast.parse(f"dummy({args_str})")
        call_node = tree.body[0].value
        
        args_dict = {}
        # Extract keyword arguments
        for kw in call_node.keywords:
            args_dict[kw.arg] = ast.literal_eval(kw.value)
            
        # Extract positional arguments
        pos_args = []
        for arg in call_node.args:
            pos_args.append(ast.literal_eval(arg))
            
        return args_dict, pos_args
    except Exception:
        return None, None


def parse_function_args(args_str, schema):
    """
    Parse arguments string into a dictionary, matching the schema.
    """
    args_str = args_str.strip()
    if not args_str:
        return {}

    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    prop_names = list(props.keys())

    # 1. Try AST parsing first (handles standard python syntax perfectly)
    ast_args, pos_args = parse_with_ast(args_str)
    if ast_args is not None:
        args = {}
        for k, v in ast_args.items():
            # Match case-insensitively to property names
            matched_key = None
            for p in prop_names:
                if p.lower() == k.lower():
                    matched_key = p
                    break
            if matched_key:
                args[matched_key] = v
            else:
                args[k] = v
        
        # Map positional arguments to remaining/required properties
        for i, val in enumerate(pos_args):
            if i < len(prop_names):
                prop_name = prop_names[i]
                if prop_name not in args:
                    args[prop_name] = val
        
        return args

    # 2. Fall back to lenient regex-based keyword parsing
    args = {}
    kw_pattern = re.compile(r"(\w+)\s*=\s*(?:['\"](.*?)['\"]|([^,]+))", re.DOTALL)
    kw_matches = kw_pattern.findall(args_str)
    
    for k, val_quoted, val_raw in kw_matches:
        val = val_quoted if val_quoted else val_raw.strip()
        # Clean up quotes if any remain
        if val.startswith(("'", '"')) and val.endswith(("'", '"')):
            val = val[1:-1]
        
        # Match case-insensitively to property names
        matched_key = None
        for p in prop_names:
            if p.lower() == k.lower():
                matched_key = p
                break
        if matched_key:
            args[matched_key] = val
        else:
            args[k] = val

    # If we parsed keyword arguments this way, return them
    if args:
        return args

    # 3. If no keyword arguments found, treat the entire args_str as a single positional argument
    val = args_str
    if val.startswith(("'", '"')) and val.endswith(("'", '"')):
        val = val[1:-1]
        
    # Map to the first property
    if prop_names:
        args[prop_names[0]] = val
    else:
        args["command"] = val
        
    return args


def parse_response(content, tools=None):
    """
    Parse tool calls from model output. Supports:
    1. Standard <tool_call> JSON-like blocks.
    2. Python-style function calls `tool_name(...)` with keyword or positional arguments.
    """
    if "STOP DOING STUFF" in content.upper():
        return content, []

    tool_calls = []
    schema_index = _tool_schema_index(tools)

    # 1. Parse <tool_call> ... </tool_call> blocks first
    xml_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    xml_matches = xml_pattern.findall(content)

    for raw in xml_matches:
        try:
            call = json.loads(raw.strip())
            items = []
            if isinstance(call, list):
                items = call
            elif isinstance(call, dict):
                if isinstance(call.get("tool_calls"), list):
                    items = call.get("tool_calls")
                else:
                    items = [call]

            for item in items:
                if not isinstance(item, dict):
                    continue
                name = (
                    item.get("name")
                    or item.get("tool")
                    or item.get("tool_name")
                    or (item.get("function") or {}).get("name")
                )
                if not name:
                    continue
                if tools and name not in schema_index:
                    continue

                arguments = (
                    item.get("arguments")
                    or item.get("parameters")
                    or item.get("args")
                    or item.get("tool_input")
                    or item.get("input")
                )
                if "function" in item and isinstance(item["function"], dict):
                    if arguments is None:
                        arguments = item["function"].get("arguments")

                schema = schema_index.get(name, {})
                normalized_args = _normalize_tool_args(name, arguments, schema)

                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(normalized_args),
                    },
                })
        except Exception:
            continue

    # Strip XML blocks from the content
    text_cleaned = xml_pattern.sub("", content).strip()

    # 2. Parse python-style function calls
    calls = find_function_calls(text_cleaned, valid_tools=schema_index)
    
    # We will construct a cleaned string by skipping the call ranges
    last_idx = 0
    cleaned_parts = []
    
    for name, args_str, start, end in calls:
        if tools and name not in schema_index:
            # Not a valid tool, treat as normal text
            cleaned_parts.append(text_cleaned[last_idx:end])
            last_idx = end
            continue
            
        # Parse the arguments using our smart lenient parser
        schema = schema_index.get(name, {})
        parsed_args = parse_function_args(args_str, schema)
        normalized_args = _normalize_tool_args(name, parsed_args, schema)
        
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(normalized_args),
            },
        })
        
        # Append the text before this call
        cleaned_parts.append(text_cleaned[last_idx:start])
        last_idx = end
        
    cleaned_parts.append(text_cleaned[last_idx:])
    text_final = "".join(cleaned_parts).strip()
    
    # Deduplicate tool calls by function name and arguments
    seen = set()
    deduped_tool_calls = []
    for tc in tool_calls:
        key = (tc["function"]["name"], tc["function"]["arguments"])
        if key not in seen:
            seen.add(key)
            deduped_tool_calls.append(tc)

    return text_final, deduped_tool_calls



def format_tool_call_for_history(name, args):
    if not isinstance(args, dict):
        return f"{name}()"
    parts = []
    for k, v in args.items():
        if isinstance(v, str):
            escaped = v.replace("'", "\\'")
            parts.append(f"{k}='{escaped}'")
        else:
            parts.append(f"{k}={v}")
    return f"{name}({', '.join(parts)})"


def format_tool_result_for_history(tool_name, content):
    return f"[Result of {tool_name}]:\n{content}"


def extract_text_content(content):
    """Extract plain text from a message content field (string or list)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


console = logging.getLogger("proxy.console")

filelog = logging.getLogger("proxy.file")


def setup_logging(log_file="proxy.log", enable_log=True):
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")

    console.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    console.addHandler(ch)

    if enable_log:
        console.setLevel(logging.DEBUG)
        filelog.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        filelog.addHandler(fh)
    else:
        filelog.setLevel(logging.CRITICAL + 1)


def log(msg):
    """Log to both console and file."""
    console.info(msg)
    filelog.info(msg)


def logfile(msg):
    """Log to both console (with --log) and file."""
    console.debug(msg)
    filelog.debug(msg)


class ProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > 0:
            raw = self.rfile.read(length)
        else:
            raw = b""
        return raw

    def _send_json(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, chunks):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_OPTIONS(self):
        self.close_connection = True
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_GET(self):
        self.close_connection = True
        if self.path in ("/v1/models", "/v1/models/"):
            log(f"GET /v1/models -> {len(MODELS)} model(s)")
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": model_id,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "chatjimmy",
                        }
                        for model_id in MODELS
                    ],
                },
            )
        else:
            self._send_json(404, {"error": "not found"})

    def _responses_to_chat(self, req):
        """Convert an OpenResponses /v1/responses request to internal chat format."""
        model = req.get("model", DEFAULT_MODEL)
        stream = req.get("stream", False)
        tools = req.get("tools", [])
        tool_choice = req.get("tool_choice", "auto")
        instructions = req.get("instructions", "")
        messages = []

        if instructions:
            messages.append({"role": "system", "content": instructions})

        raw_input = req.get("input", "")
        if isinstance(raw_input, str):
            messages.append({"role": "user", "content": raw_input})
        elif isinstance(raw_input, list):
            for item in raw_input:
                if not isinstance(item, dict):
                    continue
                t = item.get("type")
                if t == "message":
                    role = item.get("role", "user")
                    c = item.get("content", "")
                    if isinstance(c, list):
                        texts = [p.get("text", "") for p in c if p.get("type") == "input_text"]
                        c = "\n".join(texts)
                    messages.append({"role": role, "content": c})
                elif t == "function_call_output":
                    call_id = item.get("call_id", "")
                    output = item.get("output", "")
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": output, "name": "function"})

        return model, stream, tools, tool_choice, messages

    def _chat_to_responses(self, model, message, finish_reason, usage, completion_id):
        """Convert internal chat response to OpenResponses /v1/responses format."""
        output = []
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls", [])

        if content:
            output.append({
                "id": f"msg_{uuid.uuid4().hex[:12]}",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            })
        for tc in tool_calls:
            output.append({
                "type": "function_call",
                "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "call_id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "name": tc.get("function", {}).get("name", ""),
                "arguments": tc.get("function", {}).get("arguments", "{}"),
            })

        status = "completed" if finish_reason == "stop" else "in_progress" if finish_reason == "tool_calls" else "failed"
        return {
            "id": completion_id,
            "object": "response",
            "created": int(time.time()),
            "model": model,
            "status": status,
            "output": output,
            "usage": usage,
        }

    def _responses_sse_chunks(self, resp_obj, model, finish_reason, usage, completion_id):
        """Build SSE events for OpenResponses streaming."""
        now = int(time.time())
        resp_id = completion_id
        chunks = [
            {"event": "response.created", "data": {"response": {"id": resp_id, "object": "response", "status": "in_progress"}}},
            {"event": "response.in_progress", "data": {"response": {"id": resp_id, "object": "response", "status": "in_progress"}}},
        ]
        if not finish_reason == "tool_calls" and (resp_obj.get("output") or []):
            for item in resp_obj["output"]:
                chunks.append({"event": "response.output_item.added", "data": {"item": item, "response_id": resp_id}})
                if item["type"] == "message":
                    for part in item.get("content", []):
                        chunks.append({"event": "response.content_part.added", "data": {"part": part, "response_id": resp_id}})
                        if part["type"] == "output_text":
                            chunks.append({"event": "response.output_text.delta", "data": {"delta": part["text"], "response_id": resp_id}})
                            chunks.append({"event": "response.output_text.done", "data": {"text": part["text"], "response_id": resp_id}})
                    chunks.append({"event": "response.content_part.done", "data": {"response_id": resp_id}})
                chunks.append({"event": "response.output_item.done", "data": {"response_id": resp_id}})
        chunks.append({"event": "response.completed", "data": {"response": resp_obj}})
        return chunks

    def _send_responses_sse(self, chunks):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()
        for ch in chunks:
            ev = ch.get("event")
            dt = ch.get("data")
            if ev:
                self.wfile.write(f"event: {ev}\ndata: {json.dumps(dt)}\n\n".encode())
            else:
                self.wfile.write(f"data: {json.dumps(dt)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_POST(self):
        self.close_connection = True
        is_responses = self.path in ("/v1/responses", "/v1/responses/")
        if self.path not in ("/v1/chat/completions", "/v1/chat/completions/", "/v1/responses", "/v1/responses/"):
            self._send_json(404, {"error": "not found"})
            return

        raw = self._read_body()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            log("Bad request: invalid JSON")
            self._send_json(400, {"error": "invalid JSON"})
            return

        if is_responses:
            model, stream, tools, tool_choice, messages = self._responses_to_chat(body)
        else:
            messages = body.get("messages", [])
            model = body.get("model", DEFAULT_MODEL)
            stream = body.get("stream", False)
            tools = [
                t for t in body.get("tools", [])
                if t.get("function", {}).get("name", "").lower() not in FILTERED_TOOLS
            ]
            tool_choice = body.get("tool_choice", "auto")

        last_content = extract_text_content(
            messages[-1].get("content", "") if messages else ""
        )
        last_preview = (
            last_content[:100] + "..." if len(last_content) > 100 else last_content
        )

        # Console: short summary
        log(
            f'-> model={model} msgs={len(messages)} stream={stream} tools={len(tools)} | "{last_preview}"'
        )

        # File: full incoming request
        logfile("--- INCOMING REQUEST ---")
        logfile(f"Headers: {dict(self.headers)}")
        logfile(f"Body:\n{json.dumps(body, indent=2)}")

        # ----- Build system prompt & chat messages -----
        system_prompt = ""
        chat_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = extract_text_content(msg.get("content"))

            if role == "system":
                system_prompt += content + "\n"

            elif role == "assistant" and msg.get("tool_calls"):
                parts = [content] if content else []
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args = func.get("arguments", {})
                    parts.append(format_tool_call_for_history(func.get("name", ""), args))
                chat_messages.append({"role": "assistant", "content": "\n".join(parts)})

            elif role == "tool":
                tool_name = msg.get("name", "unknown")
                chat_messages.append(
                    {
                        "role": "user",
                        "content": format_tool_result_for_history(tool_name, content),
                    }
                )

            else:
                chat_messages.append({"role": role, "content": content})

        # Build system prompt with compact tool definitions
        # OpenCode sends a huge system prompt designed for large models (GPT-4, Claude).
        # The small quantized Llama 3.1 8B can't handle it — the limit is ~25K chars.
        # When the total exceeds budget, replace the bloated prompt with a minimal one
        # so tool definitions and conversation history aren't garbled by truncation.
        MAX_SYSTEM_PROMPT = 24000
        MINI_SYSTEM = "You are a helpful AI assistant with tool access. Only use tools listed below. Do not invent tools, actions, or capabilities. Be concise."

        full_system_prompt = system_prompt.strip()
        if tools:
            full_system_prompt += format_tools_for_prompt(tools, tool_choice)

        if len(full_system_prompt) > MAX_SYSTEM_PROMPT:
            logfile(
                f"WARNING: system prompt is {len(full_system_prompt)} chars "
                f"(limit {MAX_SYSTEM_PROMPT}), replacing with minimal prompt"
            )
            if tools:
                full_system_prompt = MINI_SYSTEM + format_tools_for_prompt(tools, tool_choice)
            else:
                full_system_prompt = MINI_SYSTEM

        jimmy_payload = {
            "messages": chat_messages,
            "chatOptions": {
                "selectedModel": MODELS.get(model, model),
                "systemPrompt": full_system_prompt,
                "topK": 8,
            },
            "attachment": None,
        }

        # File: translated payload
        logfile("--- TRANSLATED PAYLOAD ---")
        logfile(f"{json.dumps(jimmy_payload, indent=2)}")

        # Forward to chatjimmy
        upstream_start = time.time()
        try:
            req = urllib.request.Request(
                UPSTREAM_URL,
                data=json.dumps(jimmy_payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "*/*",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Origin": "https://chatjimmy.ai",
                    "Referer": "https://chatjimmy.ai/",
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/148.0.0.0 Safari/537.36",
                    "sec-ch-ua": '"Not/A)Brand";v="99", "Chromium";v="148"',
                    "sec-ch-ua-mobile": "?0",
                    "sec-ch-ua-platform": '"Linux"',
                    "sec-fetch-dest": "empty",
                    "sec-fetch-mode": "cors",
                    "sec-fetch-site": "same-origin",
                    "priority": "u=1, i",
                },
            )
            ctx = ssl.create_default_context()
            resp = urllib.request.urlopen(req, timeout=120, context=ctx)
            raw_response = resp.read().decode("utf-8")
            elapsed = time.time() - upstream_start
        except Exception as e:
            elapsed = time.time() - upstream_start
            log(f"<- FAILED {elapsed:.2f}s | {e}")
            logfile(f"Upstream error: {e}")
            self._send_json(502, {"error": f"upstream error: {str(e)}"})
            return

        # File: raw upstream response
        logfile("--- RAW UPSTREAM RESPONSE ---")
        logfile(raw_response)

        # Strip stats, parse usage
        content = re.sub(
            r"<\|stats\|>.*?<\|/stats\|>", "", raw_response, flags=re.DOTALL
        ).strip()
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        stats_match = re.search(
            r"<\|stats\|>(.*?)<\|/stats\|>", raw_response, re.DOTALL
        )
        if stats_match:
            try:
                stats = json.loads(stats_match.group(1))
                usage["prompt_tokens"] = stats.get("prefill_tokens", 0)
                usage["completion_tokens"] = stats.get("decode_tokens", 0)
                usage["total_tokens"] = stats.get("total_tokens", 0)
            except json.JSONDecodeError:
                pass

        # ----- Parse model's JSON response -----
        text_content, tool_calls_parsed = parse_response(content, tools)

        if tool_calls_parsed:
            finish_reason = "tool_calls"
            message = {
                "role": "assistant",
                "content": text_content if text_content else None,
                "tool_calls": tool_calls_parsed,
            }
            tc_names = [tc["function"]["name"] for tc in tool_calls_parsed]
            reply_preview = f"[tool_calls: {', '.join(tc_names)}]"
        else:
            finish_reason = "stop"
            displayed = text_content or content
            message = {"role": "assistant", "content": displayed}
            reply_preview = displayed[:100] + "..." if len(displayed) > 100 else displayed or "(empty)"

        log(f'<- {elapsed:.2f}s {usage["total_tokens"]}tok | "{reply_preview}"')

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if is_responses:
            resp_obj = self._chat_to_responses(model, message, finish_reason, usage, completion_id)
            if stream:
                chunks = self._responses_sse_chunks(resp_obj, model, finish_reason, usage, completion_id)
                self._send_responses_sse(chunks)
            else:
                self._send_json(200, resp_obj)

            logfile("--- OUTGOING RESPONSE ---")
            logfile(json.dumps(resp_obj, indent=2))
            logfile("---")
            return

        if stream:
            now = int(time.time())
            if tool_calls_parsed:
                chunks = [
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": now,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": ""},
                                "finish_reason": None,
                            }
                        ],
                    },
                ]
                if text_content:
                    chunks.append(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": now,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": text_content},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                for i, tc in enumerate(tool_calls_parsed):
                    chunks.append(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": now,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": i,
                                                "id": tc["id"],
                                                "type": "function",
                                                "function": {
                                                    "name": tc["function"]["name"],
                                                    "arguments": tc["function"][
                                                        "arguments"
                                                    ],
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                chunks.append(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": now,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                    }
                )
            else:
                chunks = [
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": now,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": now,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text_content or content},
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": now,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    },
                ]
            self._send_sse(chunks)
        else:
            openai_response = {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {"index": 0, "message": message, "finish_reason": finish_reason}
                ],
                "usage": usage,
            }
            self._send_json(200, openai_response)

        logfile("--- OUTGOING RESPONSE ---")
        if stream:
            for c in chunks:
                logfile(json.dumps(c))
        else:
            logfile(json.dumps(openai_response, indent=2))
        logfile("---")


def main():
    parser = argparse.ArgumentParser(description="ChatJimmy -> OpenAI proxy")
    parser.add_argument("--port", type=int, default=4100, help="Port to listen on")
    parser.add_argument("--log", action="store_true", help="Enable file logging")
    parser.add_argument(
        "--log-file",
        type=str,
        default="proxy.log",
        help="Log file path (requires --log)",
    )
    args = parser.parse_args()

    setup_logging(args.log_file, enable_log=args.log)
    log(f"Proxy listening on http://localhost:{args.port}/v1 -> {UPSTREAM_URL}")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
