"""A test client that drives the gateway over real pipes.

The gateway runs in a thread inside the test process, but everything around
it is real: real os.pipe() byte streams for the client side, real child
processes for the upstreams. Only the thread boundary is synthetic, and it is
the one part that cannot hide a framing or encoding bug.
"""

import json
import os
import queue
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cloakllm_gateway import config as config_mod, log  # noqa: E402
from cloakllm_gateway.gateway import Gateway  # noqa: E402

FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_upstream.py")
PROTOCOL = "2025-06-18"
TIMEOUT = 20


class Timeout(Exception):
    pass


class GatewayHarness:
    def __init__(self, upstreams, **options):
        specs = []
        for name, opts in upstreams.items():
            args = [FAKE, "--name", name]
            if opts.get("protocol"):
                args += ["--protocol", opts["protocol"]]
            if opts.get("noisy_stderr"):
                args.append("--noisy-stderr")
            if opts.get("tools_only"):
                args.append("--tools-only")
            if opts.get("command"):
                specs.append(config_mod.UpstreamSpec(
                    name, opts["command"], opts.get("args", [])))
            else:
                specs.append(config_mod.UpstreamSpec(name, sys.executable, args))
        self.config = config_mod.GatewayConfig(specs, **options)

        r_in, w_in = os.pipe()
        r_out, w_out = os.pipe()
        self._to_gateway = os.fdopen(w_in, "wb", 0)
        self._from_gateway = os.fdopen(r_out, "rb", 0)
        gw_stdin = os.fdopen(r_in, "rb", 0)
        gw_stdout = os.fdopen(w_out, "wb", 0)

        self.gateway = Gateway(self.config, gw_stdin, gw_stdout)
        self.exit_code = None
        self._inbox = queue.Queue()
        self._held = []
        self._next_id = 0

        self._server = threading.Thread(target=self._run, daemon=True)
        self._server.start()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _run(self):
        self.exit_code = self.gateway.run()

    def _read_loop(self):
        while True:
            line = self._from_gateway.readline()
            if not line:
                self._inbox.put(None)
                return
            line = line.strip()
            if not line:
                continue
            try:
                self._inbox.put(json.loads(line.decode("utf-8")))
            except (ValueError, UnicodeDecodeError):
                # Anything unparseable on stdout is a protocol corruption --
                # most likely a diagnostic that escaped onto the wrong stream.
                # Surface it as a message so a test can assert on it instead
                # of the reader thread dying and the test hanging.
                self._inbox.put({"__raw__": line.decode("utf-8", "replace")})

    # ------------------------------------------------------------- sending

    def send(self, msg):
        data = json.dumps(msg, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._to_gateway.write(data + b"\n")
        self._to_gateway.flush()

    def notify(self, method, params=None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)

    def request(self, method, params=None, msg_id=None, timeout=TIMEOUT):
        """Send a request and return its response, holding anything else."""
        if msg_id is None:
            self._next_id += 1
            msg_id = self._next_id
        msg = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)
        return self.await_response(msg_id, timeout=timeout)

    # ------------------------------------------------------------ receiving

    def _take(self, predicate):
        for i, msg in enumerate(self._held):
            if predicate(msg):
                return self._held.pop(i)
        return None

    def _next(self, predicate, timeout):
        found = self._take(predicate)
        if found is not None:
            return found
        deadline = timeout
        while True:
            try:
                msg = self._inbox.get(timeout=deadline)
            except queue.Empty:
                raise Timeout("nothing matched within %ss; held: %r"
                              % (timeout, self._held)) from None
            if msg is None:
                raise Timeout("gateway closed its output; held: %r" % (self._held,))
            if predicate(msg):
                return msg
            self._held.append(msg)

    def await_response(self, msg_id, timeout=TIMEOUT):
        return self._next(lambda m: m.get("id") == msg_id and "method" not in m,
                          timeout)

    def await_request(self, method, timeout=TIMEOUT):
        return self._next(lambda m: m.get("method") == method and m.get("id") is not None,
                          timeout)

    def await_notification(self, method, timeout=TIMEOUT):
        return self._next(lambda m: m.get("method") == method and m.get("id") is None,
                          timeout)

    def drain(self, seconds=0.4):
        """Collect whatever arrives in a quiet window. For negative checks."""
        deadline = seconds
        while True:
            try:
                msg = self._inbox.get(timeout=deadline)
            except queue.Empty:
                break
            if msg is None:
                break
            self._held.append(msg)
        return list(self._held)

    # ------------------------------------------------------------ lifecycle

    def initialize(self, protocol=PROTOCOL, capabilities=None, timeout=TIMEOUT):
        response = self.request("initialize", {
            "protocolVersion": protocol,
            "capabilities": capabilities if capabilities is not None else {"roots": {}},
            "clientInfo": {"name": "harness", "version": "1.0"},
        }, timeout=timeout)
        if "result" in response:
            self.notify("notifications/initialized", {})
        return response

    def close(self):
        try:
            self._to_gateway.close()
        except OSError:
            pass
        self._server.join(timeout=10)


@pytest.fixture
def harness():
    made = []

    def build(upstreams=None, **options):
        h = GatewayHarness(upstreams or {"fs": {}}, **options)
        made.append(h)
        return h

    log.set_level("silent")
    yield build
    for h in made:
        h.close()
        h.gateway.shutdown()
