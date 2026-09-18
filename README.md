# CloakLLM MCP Gateway

A sanitizing proxy that sits between an MCP client (Claude Desktop, Cursor, an
IDE assistant) and the MCP servers it talks to.

MCP servers are a live exfiltration path that nobody watches. A filesystem
server returns file contents; a database server returns rows; an API server
returns customer records — and all of it goes straight into the model's
context, unexamined, on its way to a provider. This is the surface where
CloakLLM's core mechanic applies *completely*, because MCP has a return path:

```
real MCP server --result--> [GATEWAY: sanitize] --> model sees [EMAIL_0]
model --tool call w/ [EMAIL_0]--> [GATEWAY: desanitize] --> real server gets the real value
```

The model never sees the PII; the downstream tool still works.

---

## Status: M0 — transparent proxy. No sanitization yet.

This is an early, unreleased repo. **Today it proxies; it does not protect.**
Do not deploy it expecting PII protection.

| Milestone | State |
|---|---|
| **M0** — transparent proxy, zero logic | **done** |
| M1 — sanitize tool results | not started |
| M2 — desanitize tool arguments | not started |
| M3 — deny-by-default walker + tripwire (**release gate**) | not started |
| M4 — hash-chained audit log | not started |

M0 exists so that when M1 starts rewriting payloads there is a proven-transparent
baseline to diff against: from here on, any difference the client can observe is
one a later milestone made on purpose, not plumbing damage.

Transparency is measured, not asserted. `tests/acceptance/m0_real_server.py`
runs the same session twice against the real
`@modelcontextprotocol/server-filesystem` — once directly, once through the
gateway — and diffs the transcripts, including every tool's full `inputSchema`
byte for byte.

---

## Honest limits

These are properties of the design, not gaps to be closed later.

1. **This is not enforcement.** A user can edit their own MCP config and remove
   the gateway. Forcing traffic through anything is a network property — an
   egress chokepoint — never a protocol property. Do not sell it as enforcement.
2. **It covers MCP traffic only.** Not the prompt a person types, not direct
   API calls, not an IDE assistant's own channel. It is a companion to the
   [CloakLLM Guard](https://github.com/cloakllm/cloakllm-guard) extension and
   the [SDK](https://github.com/cloakllm/cloakllm-py), not a replacement.
3. **Tokenized context degrades some tasks.** If the model genuinely needs a
   literal value to reason — matching, formatting, deduplication — tokenizing
   it hurts. Same trade-off the SDK has always had; per-category opt-out.
4. **It does not solve prompt injection via tool results.** Adjacent problem,
   different mechanism. No coverage is implied.
5. **Latency lands on every tool call.** A detection pass over a large file
   result is not free. Measured from M1 onward.

One more, specific to M0: **the gateway will not bridge two upstreams that
negotiate different MCP protocol versions.** It refuses the handshake and says
which upstream diverged, rather than quietly translating between dialects.

---

## Install and run

```bash
pip install -e ".[dev]"
```

Write a config naming the servers to put behind the gateway
(`cloakllm-gateway.example.json` is a starting point):

```json
{
  "upstreams": {
    "fs": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
    }
  }
}
```

Check it without serving:

```bash
cloakllm-gateway --config ./cloakllm-gateway.json --check
```

Then point your MCP client at the gateway instead of at the servers directly.
In `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "cloakllm-gateway": {
      "command": "cloakllm-gateway",
      "args": ["--config", "/absolute/path/to/cloakllm-gateway.json"]
    }
  }
}
```

**The config names processes this gateway will execute.** Treat it exactly as
you treat your client's own MCP config: it must come from a trusted location on
disk. Nothing an upstream server says can influence what gets spawned.

### Namespacing

The gateway presents many servers as one, so tool and prompt names are prefixed
with the upstream they came from, joined by `__`:

```
read_text_file   ->   fs__read_text_file
```

Upstream names may not contain `_`, which is what makes splitting on the *first*
`__` unambiguous when a tool's own name contains one: `fs__read__all` can only
be `read__all` on `fs`.

Resource URIs are **not** namespaced — a URI is meaningful to the server that
issued it and is shown to the user, so rewriting it would be a visible lie.
Resources are routed by an index built from `resources/list` instead.

## Tests

```bash
python -m pytest tests/ -q
```

The unit suite drives the gateway over real `os.pipe()` byte streams with real
child processes behind it. Only the thread boundary is synthetic — deliberately,
because a proxy is almost all plumbing and plumbing is exactly what a mock will
not test.

For the acceptance run against a real third-party server:

```bash
cd tests/acceptance
npm install @modelcontextprotocol/server-filesystem
python m0_real_server.py
```

## Design notes

Three decisions in `gateway.py` outlive M0:

- **Messages are plain dicts, never typed models.** A proxy that parses into a
  schema silently drops every field the schema does not know, and MCP grows new
  fields constantly. The gateway touches only keys it has a reason to touch.
- **Every id is remapped.** Forwarding the client's id verbatim would work
  today, but remapping gives each in-flight call somewhere to hang per-request
  state — which token map, which tool, which session — and that is what M1 and
  M2 need.
- **Unknown methods are refused, not guessed.** An unrecognised method could
  carry PII in either direction and there is no principled way to pick an
  upstream for it.

## Licence

MIT. Part of [CloakLLM](https://github.com/cloakllm).
