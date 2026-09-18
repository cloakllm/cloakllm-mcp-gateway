"""One upstream MCP server, spoken to over stdio.

Threads rather than asyncio, for two reasons. Windows cannot attach asyncio
to an inherited stdin pipe (ProactorEventLoop has no connect_read_pipe), so
the client side would need a thread anyway; and at M1 the sanitizer is
synchronous CPU work, where one thread per upstream gives the isolation we
want -- a slow scan on a large file result stalls that upstream and nothing
else.
"""

import os
import subprocess
import threading

from . import jsonrpc, log


class UpstreamError(Exception):
    """The upstream failed to start, died, or did not answer in time."""


class UpstreamRequestError(Exception):
    """The upstream answered with a JSON-RPC error."""

    def __init__(self, code, message, data=None):
        super().__init__("%s (code %s)" % (message, code))
        self.code = code
        self.rpc_message = message
        self.data = data


class Upstream:
    """A child process speaking MCP over stdin/stdout.

    on_message is called from the reader thread for every message that is not
    a response to a request this object originated. Responses to the
    gateway's own requests are consumed here and handed back to the caller of
    request().
    """

    def __init__(self, spec, on_message=None, on_exit=None):
        self.spec = spec
        self.name = spec.name
        self._on_message = on_message
        self._on_exit = on_exit

        self._proc = None
        self._writer = None
        self._reader_thread = None
        self._stderr_thread = None

        self._lock = threading.Lock()
        self._pending = {}          # our id -> _Pending
        self._next_id = 0
        self._alive = False

        # Filled in by the handshake, read by the gateway when it merges.
        self.protocol_version = None
        self.capabilities = {}
        self.server_info = {}
        self.instructions = None

    # ------------------------------------------------------------ lifecycle

    def start(self):
        # The child inherits our environment so that PATH-dependent launchers
        # (npx, uvx) work the way they do in a normal MCP client config; the
        # config's env overlays it.
        env = dict(os.environ)
        env.update(self.spec.env)

        argv = [self.spec.command] + self.spec.args
        log.debug("%s: spawning %s" % (self.name, " ".join(argv)))
        try:
            self._proc = subprocess.Popen(  # noqa: S603 - argv from trusted config
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.spec.cwd,
                env=env,
                bufsize=0,
            )
        except OSError as exc:
            raise UpstreamError("cannot start upstream %r (%s): %s"
                                % (self.name, self.spec.command, exc)) from exc

        self._alive = True
        self._writer = jsonrpc.MessageWriter(self._proc.stdin)
        self._reader_thread = threading.Thread(
            target=self._read_loop, name="up-%s-out" % self.name, daemon=True)
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(
            target=self._stderr_loop, name="up-%s-err" % self.name, daemon=True)
        self._stderr_thread.start()

    def stop(self):
        self._alive = False
        if self._writer is not None:
            self._writer.close()
        proc = self._proc
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        except OSError:
            pass
        # Fail every request still waiting, or their callers block forever.
        self._fail_all_pending("upstream %r stopped" % self.name)

    @property
    def alive(self):
        return self._alive

    # --------------------------------------------------------------- output

    def send(self, msg):
        """Forward a message as-is. Returns False if the upstream is gone."""
        if self._writer is None:
            return False
        return self._writer.write(msg)

    def request(self, method, params=None, timeout=None):
        """Send a request the gateway itself originated and wait for it.

        The id space is disjoint from anything the client uses ("cloakllm-gw-"
        prefixed strings), so a gateway request can never be confused with a
        forwarded one even if a client happens to pick the same integer.
        """
        if not self._alive:
            raise UpstreamError("upstream %r is not running" % self.name)

        pending = _Pending()
        with self._lock:
            self._next_id += 1
            req_id = "cloakllm-gw-%d" % self._next_id
            self._pending[req_id] = pending

        msg = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            msg["params"] = params

        if not self.send(msg):
            with self._lock:
                self._pending.pop(req_id, None)
            raise UpstreamError("upstream %r closed its input" % self.name)

        if not pending.event.wait(timeout):
            with self._lock:
                self._pending.pop(req_id, None)
            # Tell the upstream to stop working on it; it may still be running.
            self.send({"jsonrpc": "2.0", "method": "notifications/cancelled",
                       "params": {"requestId": req_id, "reason": "gateway timeout"}})
            raise UpstreamError("upstream %r did not answer %s within %ss"
                                % (self.name, method, timeout))

        if pending.error is not None:
            err = pending.error
            raise UpstreamRequestError(err.get("code"), err.get("message", ""),
                                       err.get("data"))
        if pending.failure is not None:
            raise UpstreamError(pending.failure)
        return pending.result

    def notify(self, method, params=None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        return self.send(msg)

    # ---------------------------------------------------------------- input

    def _read_loop(self):
        stream = self._proc.stdout
        while True:
            try:
                msg = jsonrpc.read_message(stream)
            except jsonrpc.MessageTooLarge as exc:
                log.error("%s: %s -- message dropped" % (self.name, exc))
                continue
            except jsonrpc.MalformedMessage as exc:
                log.error("%s: unparseable message dropped: %s" % (self.name, exc))
                continue
            except (OSError, ValueError):
                break
            if msg is None:
                break
            try:
                self._dispatch(msg)
            except Exception as exc:  # noqa: BLE001 - a reader thread must not die
                log.error("%s: error handling message: %r" % (self.name, exc))

        self._alive = False
        code = self._proc.poll() if self._proc else None
        log.info("%s: upstream exited (code %s)" % (self.name, code))
        self._fail_all_pending("upstream %r exited" % self.name)
        if self._on_exit is not None:
            try:
                self._on_exit(self)
            except Exception as exc:  # noqa: BLE001
                log.error("%s: exit handler failed: %r" % (self.name, exc))

    def _dispatch(self, msg):
        msg_id = msg.get("id")
        if jsonrpc.is_response(msg) and isinstance(msg_id, str):
            with self._lock:
                pending = self._pending.pop(msg_id, None)
            if pending is not None:
                pending.result = msg.get("result")
                pending.error = msg.get("error")
                pending.event.set()
                return
        if self._on_message is not None:
            self._on_message(self, msg)

    def _fail_all_pending(self, reason):
        with self._lock:
            waiting = list(self._pending.values())
            self._pending.clear()
        for pending in waiting:
            pending.failure = reason
            pending.event.set()

    def _stderr_loop(self):
        # An upstream's stderr is its diagnostic channel and often the only
        # clue why a handshake failed, so it is surfaced rather than dropped.
        # It must never reach our stdout, which carries the protocol.
        stream = self._proc.stderr
        while True:
            try:
                line = stream.readline()
            except (OSError, ValueError):
                break
            if not line:
                break
            text = line.decode("utf-8", "replace").rstrip()
            if text:
                log.debug("%s[stderr] %s" % (self.name, text))


class _Pending:
    __slots__ = ("event", "result", "error", "failure")

    def __init__(self):
        self.event = threading.Event()
        self.result = None
        self.error = None
        self.failure = None
