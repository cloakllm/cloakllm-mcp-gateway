"""A minimal MCP server over stdio, for testing the gateway against.

Not a mock object -- a real child process speaking the real framing over real
pipes. The distinction matters here: the extension's M4 bug (chrome.storage
used with no permission declared) got through because every test ran against
an injected in-memory seam, so the mock shared the implementation's blind
spot. A proxy is almost all plumbing, and plumbing is exactly what a mock
will not test.

Deliberately awkward in the ways a real server is: it paginates, it carries
fields the gateway has never heard of, it asks the client questions, and it
can be made to hang.
"""

import argparse
import json
import sys
import threading

PROTOCOL = "2025-06-18"


class FakeUpstream:
    def __init__(self, name, protocol=PROTOCOL, tools_only=False):
        self.name = name
        self.protocol = protocol
        # The real filesystem server declares tools and nothing else, and
        # answers -32601 for the rest. That asymmetry is what exposed the
        # gateway synthesising empty lists, so it has to be reproducible here.
        self.tools_only = tools_only
        self._stdout = sys.stdout.buffer
        self._lock = threading.Lock()
        self._cancelled = set()
        self._pending_sampling = {}
        self._next_id = 0

    # ---------------------------------------------------------------- wire

    def send(self, obj):
        data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        with self._lock:
            self._stdout.write(data + b"\n")
            self._stdout.flush()

    def reply(self, msg_id, result):
        self.send({"jsonrpc": "2.0", "id": msg_id, "result": result})

    def fail(self, msg_id, code, message):
        self.send({"jsonrpc": "2.0", "id": msg_id,
                   "error": {"code": code, "message": message}})

    def run(self):
        stream = sys.stdin.buffer
        while True:
            line = stream.readline()
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line.decode("utf-8"))
            threading.Thread(target=self._handle, args=(msg,), daemon=True).start()

    # ------------------------------------------------------------ handlers

    def _handle(self, msg):
        method = msg.get("method")
        msg_id = msg.get("id")
        params = msg.get("params") or {}

        if method is None:
            self._on_response(msg)
            return
        if msg_id is None:
            self._on_notification(method, params)
            return

        if self.tools_only and method.split("/")[0] in ("resources", "prompts",
                                                        "completion", "logging"):
            self.fail(msg_id, -32601, "method not found: %s" % method)
            return
        handler = getattr(self, "_do_" + method.replace("/", "_"), None)
        if handler is None:
            self.fail(msg_id, -32601, "method not found: %s" % method)
            return
        handler(msg_id, params)

    def _on_notification(self, method, params):
        if method == "notifications/cancelled":
            self._cancelled.add(_key(params.get("requestId")))

    def _on_response(self, msg):
        event_result = self._pending_sampling.get(msg.get("id"))
        if event_result is not None:
            event_result[1] = msg
            event_result[0].set()

    def _do_initialize(self, msg_id, params):
        capabilities = {"tools": {"listChanged": True}}
        if not self.tools_only:
            capabilities.update({
                "resources": {"subscribe": True, "listChanged": True},
                "prompts": {},
                "logging": {},
            })
        self.reply(msg_id, {
            "protocolVersion": self.protocol,
            "capabilities": capabilities,
            "serverInfo": {"name": "fake-%s" % self.name, "version": "9.9.9"},
            "instructions": "Instructions from %s." % self.name,
        })

    def _do_ping(self, msg_id, _params):
        self.reply(msg_id, {})

    # Two pages, so the gateway has to follow nextCursor rather than taking
    # the first page and silently losing the rest.
    def _do_tools_list(self, msg_id, params):
        cursor = params.get("cursor")
        if not cursor:
            self.reply(msg_id, {
                "tools": [{
                    "name": "echo",
                    "description": "Echo the arguments back.",
                    "inputSchema": {"type": "object"},
                    # A field no version of the gateway knows about. It must
                    # arrive at the client unchanged.
                    "x-vendor-extension": {"deep": ["preserve", "me"]},
                }],
                "nextCursor": "page2",
            })
            return
        self.reply(msg_id, {"tools": [
            {"name": "ask_client", "description": "Ask the client to sample.",
             "inputSchema": {"type": "object"}},
            {"name": "slow", "description": "Never answers.",
             "inputSchema": {"type": "object"}},
            {"name": "boom", "description": "Always errors.",
             "inputSchema": {"type": "object"}},
        ]})

    def _do_tools_call(self, msg_id, params):
        name = params.get("name")
        args = params.get("arguments") or {}

        if name == "echo":
            self.reply(msg_id, {
                "content": [{"type": "text",
                             "text": "%s echoes %s" % (self.name, json.dumps(args, sort_keys=True))}],
                "isError": False,
                "x-result-extension": "preserve me too",
            })
            return
        if name == "ask_client":
            self.reply(msg_id, {"content": [
                {"type": "text", "text": json.dumps(self._ask_client())}]})
            return
        if name == "slow":
            # Answers only once a cancellation lands, so a test can prove the
            # notification was routed and rewritten correctly.
            for _ in range(200):
                if _key(msg_id) in self._cancelled:
                    self.reply(msg_id, {"content": [
                        {"type": "text", "text": "cancelled"}]})
                    return
                threading.Event().wait(0.02)
            self.reply(msg_id, {"content": [{"type": "text", "text": "timed out"}]})
            return
        if name == "boom":
            self.fail(msg_id, -32000, "boom from %s" % self.name)
            return
        self.fail(msg_id, -32602, "unknown tool: %s" % name)

    def _ask_client(self):
        self._next_id += 1
        req_id = "srv-%d" % self._next_id
        slot = [threading.Event(), None]
        self._pending_sampling[req_id] = slot
        self.send({"jsonrpc": "2.0", "id": req_id, "method": "sampling/createMessage",
                   "params": {"messages": [], "maxTokens": 1, "from": self.name}})
        if not slot[0].wait(10):
            return {"error": "client never answered"}
        return (slot[1] or {}).get("result")

    def _do_resources_list(self, msg_id, _params):
        self.reply(msg_id, {"resources": [
            {"uri": "file:///%s/notes.txt" % self.name, "name": "notes",
             "mimeType": "text/plain"}]})

    def _do_resources_templates_list(self, msg_id, _params):
        self.reply(msg_id, {"resourceTemplates": [
            {"uriTemplate": "file:///%s/{path}" % self.name, "name": "any file"}]})

    def _do_resources_read(self, msg_id, params):
        self.reply(msg_id, {"contents": [
            {"uri": params.get("uri"), "mimeType": "text/plain",
             "text": "contents of %s from %s" % (params.get("uri"), self.name)}]})

    def _do_prompts_list(self, msg_id, _params):
        self.reply(msg_id, {"prompts": [
            {"name": "greet", "description": "Say hello.",
             "arguments": [{"name": "who", "required": True}]}]})

    def _do_prompts_get(self, msg_id, params):
        self.reply(msg_id, {"description": "greeting", "messages": [
            {"role": "user", "content": {
                "type": "text",
                "text": "hello %s from %s" % (
                    (params.get("arguments") or {}).get("who"), self.name)}}]})

    def _do_logging_setLevel(self, msg_id, _params):  # noqa: N802 - MCP method name
        self.reply(msg_id, {})

    def _do_completion_complete(self, msg_id, params):
        self.reply(msg_id, {"completion": {
            "values": ["completed by %s for %s"
                       % (self.name, json.dumps(params.get("ref"), sort_keys=True))],
            "hasMore": False}})


def _key(value):
    return (type(value).__name__, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="fake")
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument("--noisy-stderr", action="store_true",
                        help="write to stderr, to prove it never reaches stdout")
    parser.add_argument("--tools-only", action="store_true",
                        help="declare tools only, like the real filesystem server")
    args = parser.parse_args()

    if args.noisy_stderr:
        sys.stderr.write("fake upstream %s says hello on stderr\n" % args.name)
        sys.stderr.flush()

    FakeUpstream(args.name, args.protocol, args.tools_only).run()


if __name__ == "__main__":
    main()
