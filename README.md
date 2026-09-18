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

## Status: M2 — the round trip works. Not yet released.

This is an early, unreleased repo. **The full loop is closed: the model sees
`[EMAIL_0]`, the downstream tool receives the real address.** The remaining
milestones are the leak-probe gate and the audit chain, and the release gate is
M3 — so do not deploy this in anger yet.

| Milestone | State |
|---|---|
| **M0** — transparent proxy, zero logic | **done** |
| **M1** — sanitize tool results | **done** |
| **M2** — desanitize tool arguments | **done** |
| M3 — deny-by-default walker + tripwire (**release gate**) | not started |
| M4 — hash-chained audit log | not started |

The round trip is proved against a real server, on disk rather than through the
gateway. The acceptance run has the model write a file whose content is the
token `[EMAIL_0]`, then opens that file directly:

```
model sent to the tool:  Contact: [EMAIL_0]
what landed on disk:     Contact: marie.dubois@example-eu.fr
```

Reading it back through the gateway would have proved nothing — a result is
re-sanitized on the way out, so "restored correctly" and "never restored at
all" look identical from there.

Behaviour is measured against a real third-party server, not asserted.
`tests/acceptance/real_server.py` runs the same session three ways — straight at
`@modelcontextprotocol/server-filesystem`, through the gateway with
sanitization off, and through it with sanitization on — and diffs the
transcripts, including every tool's full `inputSchema` byte for byte.

Running all three is what makes either claim checkable. The transparent pass
proves the gateway changes nothing it did not mean to; the protecting pass
proves the one thing it changes is the PII. Neither is worth much alone — a
gateway that mangled everything would pass the second, and one that did nothing
would pass the first.

### What is sanitized, and what deliberately is not

**Towards the model** — `tools/call`, `resources/read`, `prompts/get`,
`resources/list`, `resources/templates/list`, and `sampling/createMessage`
requests an upstream makes of the client. Results **and** JSON-RPC error
bodies: the real filesystem server puts the requested path in its "file not
found" message, so a failed call leaks exactly what a successful one would.

**Towards the upstream** — `tools/call` arguments and sampling answers have
their tokens restored, so the tool receives real values.

**Not touched:** `tools/list` and `prompts/list`. Those carry schemas.
Rewriting a description or an enum inside an `inputSchema` corrupts the
contract the model calls against, for no privacy gain. `resources/list` is the
opposite case and *is* sanitized — a directory listing is data, and filenames
carry names and addresses.

The walker **sanitizes every string by default** and skips only `type`,
`mimeType`, `progressToken`, `data` and `blob` — protocol discriminators,
correlation identities and base64 binary, whose exact bytes must survive. A
field this gateway has never heard of is therefore sanitized, not ignored.
Every skip is counted, so M3's tripwire has an explicit list of surfaces.

### Two things worth knowing about the round trip

**A hostile upstream cannot harvest values with a fake token.** A third-party
server can return text containing a literal `[EMAIL_0]`, hoping the model
quotes it into a later call so the gateway substitutes a real address into a
payload bound for the attacker. The SDK's token-injection escaping rewrites
brackets in upstream content to fullwidth ones, so what the model sees is not a
token and never resolves. There is a test for it.

**PII the model writes out itself is reported, not blocked.** A restored token
is the system working; a raw address the model produced is something else. The
gateway logs it and does nothing more, because it cannot tell a filesystem
server on the same machine from a third-party server that happens to speak MCP
— only the operator can. Per-upstream trust is the design that would turn this
into enforcement, and it is not built.

### Detection defaults

The SDK enables every NER label it knows. This gateway defaults to **PERSON
only**, because a gateway tokenizes what the model then has to reason over: with
the full set, *"what is the capital of France"* reaches the model as *"the
capital of `[GPE_0]`"* and the answer is gone. `France` and `Acme Corp` identify
nobody. Widen it with `detection.ner_entity_types` if you want the SDK's
behaviour.

Note that narrowing the NER set changes **what** is tokenized, not what it
costs — the model still runs.

### Latency

Measured on one developer machine, so treat these as an order of magnitude
rather than a benchmark. Per sanitized payload:

| Payload | Median |
|---|---|
| 500 chars | ~12 ms |
| 5 KB | ~86 ms |
| 50 KB | ~900 ms |

It is superlinear, and it lands on every tool call. A tool that returns whole
files will feel it.

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
   result is not free — see the table above.
6. **Detection is not perfect, and the guarantee is only as good as it is.**
   The SDK measures roughly 97% character-level scrub on deliberately hard
   inputs. This gateway inherits exactly that, no better.

One more, specific to M0: **the gateway will not bridge two upstreams that
negotiate different MCP protocol versions.** It refuses the handshake and says
which upstream diverged, rather than quietly translating between dialects.

---

## Install and run

```bash
pip install -e ".[detection]"        # omit [detection] for regex categories only
python -m spacy download en_core_web_sm
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
  },
  "sanitize": true,
  "detection": { "detect_ip_addresses": false }
}
```

An unknown key under `detection` is rejected rather than ignored: a typo there
is the worst failure available, because the operator believes a category is on
when it is not.

On startup the gateway runs known values of every enabled category through the
real detection path and **refuses to start if any survives**. Constructing a
detector is not evidence that it detects — a spaCy model that failed to load
produces a perfectly healthy-looking object that sanitizes nothing, and the
gateway would otherwise come up announcing protection it was not providing.

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
python real_server.py
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
