"""M0 acceptance against a REAL MCP server, measured rather than eyeballed.

"Cannot tell the difference" is only a claim until you run the same session
twice -- once straight at @modelcontextprotocol/server-filesystem, once at
the same server through the gateway -- and diff the two transcripts.

The fake upstream in tests/ is a mock, so it shares the assumptions of the
code it is testing. This does not -- and it earned its keep immediately: the
empty-list-instead-of--32601 divergence was invisible to the mock, because
the mock implements every primitive and the real filesystem server does not.

Kept out of the default pytest run because it needs a node_modules. To run:

    cd tests/acceptance
    npm install @modelcontextprotocol/server-filesystem
    python m0_real_server.py
"""
import json
import os
import subprocess
import sys
import threading
import queue

HERE = os.path.dirname(os.path.abspath(__file__))
GATEWAY_REPO = os.path.dirname(os.path.dirname(HERE))
FS_SERVER = os.path.join(HERE, "node_modules", "@modelcontextprotocol",
                         "server-filesystem", "dist", "index.js")
SANDBOX = os.path.join(HERE, "sandbox")
PROTOCOL = "2025-06-18"

fails = []


def check(label, cond, detail=""):
    print("   %s  %s%s" % ("PASS" if cond else "FAIL", label,
                           ("  -- " + str(detail)[:400]) if detail and not cond else ""))
    if not cond:
        fails.append(label)


class Peer:
    """A real MCP client over real pipes to a real child process."""

    def __init__(self, argv, cwd=None):
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, cwd=cwd, bufsize=0)
        self.inbox = queue.Queue()
        self.stderr = []
        self._n = 0
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()

    def _read(self):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                self.inbox.put(None)
                return
            line = line.strip()
            if not line:
                continue
            try:
                self.inbox.put(json.loads(line.decode("utf-8")))
            except ValueError:
                self.inbox.put({"__raw__": line.decode("utf-8", "replace")})

    def _read_err(self):
        while True:
            line = self.proc.stderr.readline()
            if not line:
                return
            self.stderr.append(line.decode("utf-8", "replace").rstrip())

    def send(self, msg):
        self.proc.stdin.write(
            json.dumps(msg, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
        self.proc.stdin.flush()

    def request(self, method, params=None, timeout=60):
        self._n += 1
        msg = {"jsonrpc": "2.0", "id": self._n, "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)
        while True:
            got = self.inbox.get(timeout=timeout)
            if got is None:
                raise RuntimeError("peer closed stdout; stderr: %s" % self.stderr[-5:])
            if got.get("id") == self._n:
                return got

    def initialize(self):
        r = self.request("initialize", {
            "protocolVersion": PROTOCOL,
            "capabilities": {"roots": {"listChanged": True}},
            "clientInfo": {"name": "m0-acceptance", "version": "1.0"}})
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return r

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def setup_sandbox():
    os.makedirs(SANDBOX, exist_ok=True)
    # A planted value, so M1 has something concrete to change. At M0 it must
    # arrive verbatim; at M1 it must arrive as [EMAIL_0].
    with open(os.path.join(SANDBOX, "customer.txt"), "w", encoding="utf-8") as fh:
        fh.write("Customer marie.dubois@example-eu.fr called about card "
                 "5500 0000 0000 0004.\n")
    return os.path.join(SANDBOX, "customer.txt")


def write_config(path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({
            "upstreams": {"fs": {"command": "node", "args": [FS_SERVER, SANDBOX]}},
            "log_level": "warn",
        }, fh)


def main():
    if not os.path.isfile(FS_SERVER):
        print("The real server is not installed. From %s run:\n"
              "    npm install @modelcontextprotocol/server-filesystem" % HERE)
        return 2

    planted_file = setup_sandbox()
    cfg = os.path.join(HERE, "gw.json")
    write_config(cfg)

    print("\nM0 acceptance: @modelcontextprotocol/server-filesystem")
    print("   direct  : node %s" % os.path.basename(FS_SERVER))
    print("   gateway : python -m cloakllm_gateway -> the same server\n")

    direct = Peer(["node", FS_SERVER, SANDBOX])
    gw = Peer([sys.executable, "-m", "cloakllm_gateway", "--config", cfg],
              cwd=GATEWAY_REPO)

    try:
        print("1. handshake")
        d_init = direct.initialize()
        g_init = gw.initialize()
        check("direct server initializes", "result" in d_init, d_init)
        check("gateway initializes", "result" in g_init, g_init)
        check("same protocol version negotiated",
              d_init["result"]["protocolVersion"] == g_init["result"]["protocolVersion"],
              (d_init["result"]["protocolVersion"], g_init["result"]["protocolVersion"]))
        check("gateway identifies itself, not the upstream",
              g_init["result"]["serverInfo"]["name"] == "cloakllm-gateway",
              g_init["result"]["serverInfo"])

        print("\n2. tools/list -- the transcripts must match once de-namespaced")
        d_tools = direct.request("tools/list")["result"]["tools"]
        g_tools = gw.request("tools/list")["result"]["tools"]
        check("the real server exposes tools", len(d_tools) > 0, len(d_tools))
        check("gateway exposes the same number", len(d_tools) == len(g_tools),
              "direct %d vs gateway %d" % (len(d_tools), len(g_tools)))

        stripped = []
        for tool in g_tools:
            copy = dict(tool)
            check("tool %r is namespaced" % tool["name"],
                  tool["name"].startswith("fs__"), tool["name"])
            copy["name"] = tool["name"][len("fs__"):]
            stripped.append(copy)

        by_name_direct = {t["name"]: t for t in d_tools}
        by_name_gw = {t["name"]: t for t in stripped}
        check("the same tool names", set(by_name_direct) == set(by_name_gw),
              set(by_name_direct) ^ set(by_name_gw))

        # The strong form: every tool definition, byte for byte, including
        # the full inputSchema. This is what would catch a proxy that parsed
        # into a typed model and dropped a field on the way through.
        differing = [n for n in by_name_direct
                     if json.dumps(by_name_direct[n], sort_keys=True)
                     != json.dumps(by_name_gw.get(n), sort_keys=True)]
        check("every tool definition is byte-identical after de-namespacing",
              not differing, differing[:3])

        print("\n3. tools/call -- a real file read, through both paths")
        args = {"path": planted_file}
        d_read = direct.request("tools/call", {"name": "read_text_file", "arguments": args})
        g_read = gw.request("tools/call", {"name": "fs__read_text_file", "arguments": args})
        check("direct read succeeded", "result" in d_read, d_read)
        check("gateway read succeeded", "result" in g_read, g_read)
        check("results are byte-identical",
              json.dumps(d_read.get("result"), sort_keys=True)
              == json.dumps(g_read.get("result"), sort_keys=True),
              (d_read.get("result"), g_read.get("result")))

        text = json.dumps(g_read.get("result"))
        # M0 is transparent BY DESIGN, so the planted PII must still be here.
        # This assertion inverts at M1 and is the single clearest marker of
        # the milestone boundary.
        check("M0 baseline: planted PII passes through untouched (inverts at M1)",
              "marie.dubois@example-eu.fr" in text, text[:200])

        print("\n4. error paths and unknown tools")
        d_err = direct.request("tools/call", {
            "name": "read_text_file", "arguments": {"path": os.path.join(SANDBOX, "nope.txt")}})
        g_err = gw.request("tools/call", {
            "name": "fs__read_text_file", "arguments": {"path": os.path.join(SANDBOX, "nope.txt")}})
        check("a real error surfaces the same way through the gateway",
              json.dumps(_norm(d_err), sort_keys=True) == json.dumps(_norm(g_err), sort_keys=True),
              (_norm(d_err), _norm(g_err)))

        print("\n5. primitives this server does NOT implement")
        # The first version of this section short-circuited whenever the
        # direct server errored, so it passed without comparing anything --
        # and hid a real divergence. A negative result is only evidence if
        # the check could have gone the other way.
        for method in ("resources/list", "prompts/list", "resources/templates/list"):
            d_r = direct.request(method)
            g_r = gw.request(method)
            d_kind = "error" if "error" in d_r else "result"
            g_kind = "error" if "error" in g_r else "result"
            check("%s: gateway answers in kind (%s)" % (method, d_kind),
                  d_kind == g_kind,
                  "direct %s=%s vs gateway %s=%s"
                  % (d_kind, d_r.get(d_kind), g_kind, g_r.get(g_kind)))
            if d_kind == "error" == g_kind:
                check("%s: same error code" % method,
                      d_r["error"]["code"] == g_r["error"]["code"],
                      (d_r["error"], g_r["error"]))

        print("\n6. hygiene")
        leaked = [m for m in list(gw.inbox.queue) if "__raw__" in m]
        check("nothing non-JSON reached the client's stdin", not leaked, leaked[:2])
        check("gateway still responsive after everything",
              "result" in gw.request("ping", timeout=20))

    finally:
        gw.close()
        direct.close()

    print("\nM0 acceptance: %s"
          % ("FAILED -- " + "; ".join(fails) if fails else "all checks passed"))
    return 1 if fails else 0


def _norm(resp):
    """Compare shape and message, not ids."""
    out = {k: v for k, v in resp.items() if k != "id"}
    return out


if __name__ == "__main__":
    sys.exit(main())
