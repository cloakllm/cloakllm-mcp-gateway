"""The proxy itself: one MCP server to the client, many upstreams behind it.

M0 is deliberately transparent -- no sanitization, no audit chain. The point
of this milestone is to prove the plumbing can carry real traffic without the
client noticing, so that when M1 starts rewriting payloads there is a known
good baseline to diff against.

Three structural decisions here outlive M0:

**Every id is remapped.** A forwarded client request goes out under a
gateway-allocated id and the response is mapped back. Forwarding the client's
id verbatim would work today, but remapping gives each in-flight call a place
to hang per-request state -- which token map, which tool, which session --
and that is exactly what M1 and M2 need.

**Unknown methods are refused, not guessed.** An unrecognised method could
carry PII in either direction and there is no principled way to pick an
upstream for it. This is the LlamaIndex E1 lesson (a closed allow-list that
let unknown content types through raw) applied at the method layer: refusing
is visible, forwarding blind is not.

**serverInfo is ours, not an upstream's.** The gateway tells the client what
it is. A tool whose value is an audit trail should not impersonate something
else in its own handshake.
"""

import concurrent.futures
import itertools
import threading
import time

from . import jsonrpc, log
from .upstream import Upstream, UpstreamError, UpstreamRequestError

GATEWAY_NAME = "cloakllm-gateway"

# Versions whose message shapes this gateway has been written against. An
# unknown version is refused rather than best-guessed: a gateway that keeps
# forwarding while no longer understanding the payloads is one that has
# silently stopped protecting anything, which is worse than not being there.
SUPPORTED_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")
PREFERRED_PROTOCOL_VERSION = "2025-06-18"

NAMESPACE_SEP = "__"

# A broken upstream that always returns a nextCursor would otherwise spin here.
MAX_LIST_PAGES = 1000


class Gateway:
    def __init__(self, config, stdin, stdout):
        self.config = config
        self._stdin = stdin
        self._client = jsonrpc.MessageWriter(stdout)

        self.upstreams = {}
        for spec in config.upstreams:
            self.upstreams[spec.name] = Upstream(
                spec, on_message=self._on_upstream_message,
                on_exit=self._on_upstream_exit)

        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        # Gateway-allocated ids are strings this process minted, so they key
        # these two maps directly.
        self._from_client = {}      # gw id -> (upstream name, client id, method)
        self._from_upstream = {}    # gw id -> (upstream name, upstream id)
        # These two are keyed by ids that came from a peer, so they go through
        # _key(): a client using string ids must not alias one using integers.
        self._client_id_index = {}      # _key(client id) -> gw id
        self._upstream_id_index = {}    # (upstream name, _key(id)) -> gw id

        self._resource_owner = {}   # uri -> upstream name
        self._initialized = False
        self._protocol_version = None
        self._stopping = False

        self._workers = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(4, len(self.upstreams) * 2),
            thread_name_prefix="gw-work")

    # ------------------------------------------------------------- run loop

    def run(self):
        """Read the client until it closes. Returns a process exit code."""
        for name, up in self.upstreams.items():
            try:
                up.start()
            except UpstreamError as exc:
                log.error(str(exc))
                self.shutdown()
                return 1
            log.info("%s: started" % name)

        log.info("listening on stdio with %d upstream(s): %s"
                 % (len(self.upstreams), ", ".join(sorted(self.upstreams))))

        try:
            while True:
                try:
                    msg = jsonrpc.read_message(self._stdin)
                except jsonrpc.MessageTooLarge as exc:
                    log.error("client: %s -- message dropped" % exc)
                    continue
                except jsonrpc.MalformedMessage as exc:
                    log.error("client: unparseable message dropped: %s" % exc)
                    self._client.write(jsonrpc.error_response(
                        None, jsonrpc.PARSE_ERROR, "invalid JSON: %s" % exc))
                    continue
                except (OSError, ValueError):
                    break
                if msg is None:
                    break
                try:
                    self._on_client_message(msg)
                except Exception as exc:  # noqa: BLE001 - the loop must survive
                    log.error("error handling client message: %r" % exc)
                    if jsonrpc.is_request(msg):
                        self._client.write(jsonrpc.error_response(
                            msg["id"], jsonrpc.INTERNAL_ERROR,
                            "gateway error: %s" % exc))
        finally:
            self.shutdown()
        return 0

    def shutdown(self):
        with self._lock:
            if self._stopping:
                return
            self._stopping = True
        log.info("shutting down")
        self._workers.shutdown(wait=False)
        for up in self.upstreams.values():
            up.stop()
        self._client.close()

    # ------------------------------------------------------- client inbound

    def _on_client_message(self, msg):
        if jsonrpc.is_response(msg):
            self._forward_client_response(msg)
            return
        if jsonrpc.is_notification(msg):
            self._on_client_notification(msg)
            return
        if not jsonrpc.is_request(msg):
            log.warn("client sent a message that is neither request, "
                     "notification nor response; dropped")
            return

        method = msg.get("method")
        msg_id = msg["id"]

        if method == "initialize":
            self._workers.submit(self._guarded, msg_id, self._handle_initialize, msg)
            return
        if method == "ping":
            # A ping asks whether *this* peer is alive. Answering locally is
            # correct; fanning it out would make our liveness depend on
            # every upstream's.
            self._client.write(jsonrpc.result_response(msg_id, {}))
            return

        if not self._initialized:
            self._client.write(jsonrpc.error_response(
                msg_id, jsonrpc.INVALID_REQUEST,
                "received %r before initialize" % method))
            return

        if method in ("tools/list", "prompts/list", "resources/list",
                      "resources/templates/list"):
            self._workers.submit(self._guarded, msg_id, self._handle_list, msg)
            return
        if method == "logging/setLevel":
            self._workers.submit(self._guarded, msg_id, self._handle_set_level, msg)
            return

        target = self._route(method, msg.get("params") or {})
        if isinstance(target, dict):          # a prepared error
            self._client.write(jsonrpc.error_response(msg_id, **target))
            return
        upstream, params = target
        self._forward_client_request(upstream, msg, params)

    def _guarded(self, msg_id, fn, msg):
        """Run a blocking handler on a worker, answering the client either way."""
        try:
            fn(msg)
        except UpstreamRequestError as exc:
            self._client.write(jsonrpc.error_response(
                msg_id, exc.code if isinstance(exc.code, int) else jsonrpc.INTERNAL_ERROR,
                exc.rpc_message, exc.data))
        except UpstreamError as exc:
            self._client.write(jsonrpc.error_response(
                msg_id, jsonrpc.INTERNAL_ERROR, str(exc)))
        except Exception as exc:  # noqa: BLE001
            log.error("handler failed: %r" % exc)
            self._client.write(jsonrpc.error_response(
                msg_id, jsonrpc.INTERNAL_ERROR, "gateway error: %s" % exc))

    def _on_client_notification(self, msg):
        method = msg.get("method")
        params = msg.get("params") or {}

        if method == "notifications/initialized":
            for up in self.upstreams.values():
                up.notify(method, params)
            return
        if method == "notifications/cancelled":
            self._forward_client_cancel(params)
            return
        if method == "notifications/roots/list_changed":
            for up in self.upstreams.values():
                up.notify(method, params)
            return
        # Deny-by-default: an unrecognised notification has no defensible
        # destination, and broadcasting one to every upstream is a guess.
        log.debug("client notification %r not handled; dropped" % method)

    # ------------------------------------------------------------ handshake

    def _handle_initialize(self, msg):
        params = msg.get("params") or {}
        requested = params.get("protocolVersion")

        if requested not in SUPPORTED_PROTOCOL_VERSIONS:
            if not self.config.allow_unknown_protocol_version:
                self._client.write(jsonrpc.error_response(
                    msg["id"], jsonrpc.INVALID_PARAMS,
                    "unsupported MCP protocol version %r" % requested,
                    {"supported": list(SUPPORTED_PROTOCOL_VERSIONS)}))
                log.error("refusing protocol version %r; supported: %s"
                          % (requested, ", ".join(SUPPORTED_PROTOCOL_VERSIONS)))
                return
            log.warn("protocol version %r is not one this gateway was written "
                     "against; proceeding because allow_unknown_protocol_version "
                     "is set" % requested)

        # clientInfo is forwarded unchanged so upstreams see the real client,
        # which is what a proxy owes them. Only the response identifies us.
        results = self._fan_out(
            lambda up: up.request("initialize", params,
                                  timeout=up.spec.startup_timeout))

        failures = [(n, e) for n, (ok, e) in results.items() if not ok]
        if failures:
            detail = "; ".join("%s: %s" % (n, e) for n, e in failures)
            self._client.write(jsonrpc.error_response(
                msg["id"], jsonrpc.INTERNAL_ERROR,
                "upstream initialize failed -- %s" % detail))
            log.error("initialize failed for %d upstream(s): %s"
                      % (len(failures), detail))
            return

        versions = {}
        for name, (_, result) in results.items():
            up = self.upstreams[name]
            up.protocol_version = (result or {}).get("protocolVersion")
            up.capabilities = (result or {}).get("capabilities") or {}
            up.server_info = (result or {}).get("serverInfo") or {}
            up.instructions = (result or {}).get("instructions")
            versions.setdefault(up.protocol_version, []).append(name)

        if len(versions) > 1:
            # One version has to be presented to the client. Picking the
            # lowest would leave the gateway speaking two dialects at once
            # and quietly translating between them, which is precisely the
            # silent degradation this design refuses.
            detail = "; ".join("%s: %s" % (v, ", ".join(sorted(n)))
                               for v, n in sorted(versions.items(), key=lambda kv: str(kv[0])))
            self._client.write(jsonrpc.error_response(
                msg["id"], jsonrpc.INTERNAL_ERROR,
                "upstreams negotiated different protocol versions, which this "
                "gateway will not bridge -- %s" % detail))
            log.error("protocol version split across upstreams: %s" % detail)
            return

        negotiated = next(iter(versions)) or requested
        self._protocol_version = negotiated

        result = {
            "protocolVersion": negotiated,
            "capabilities": self._merged_capabilities(),
            "serverInfo": {"name": GATEWAY_NAME, "version": _version()},
        }
        instructions = self._merged_instructions()
        if instructions:
            result["instructions"] = instructions

        self._initialized = True
        self._client.write(jsonrpc.result_response(msg["id"], result))
        log.info("initialized: protocol %s, upstreams %s"
                 % (negotiated, ", ".join(sorted(self.upstreams))))

    def _merged_capabilities(self):
        """Advertise the union of what the upstreams can do.

        Only capabilities the gateway actually routes are advertised.
        Anything else stays off, because advertising a capability we would
        then refuse is a worse failure than not offering it.
        """
        merged = {}
        for up in self.upstreams.values():
            caps = up.capabilities or {}
            for key in ("tools", "resources", "prompts", "logging", "completions"):
                if key not in caps:
                    continue
                value = caps[key] if isinstance(caps[key], dict) else {}
                slot = merged.setdefault(key, {})
                for flag in ("listChanged", "subscribe"):
                    if value.get(flag):
                        slot[flag] = True
        return merged

    def _merged_instructions(self):
        chunks = []
        for name in sorted(self.upstreams):
            text = self.upstreams[name].instructions
            if text:
                chunks.append("## %s\n\n%s" % (name, text))
        if not chunks:
            return None
        header = ("Tools are namespaced by upstream server as "
                  "`<server>%sname`.\n" % NAMESPACE_SEP)
        return header + "\n\n".join(chunks)

    # -------------------------------------------------------------- routing

    def _route(self, method, params):
        """Pick the upstream for a request, and rewrite params for it.

        Returns (upstream, params) or a dict of error-response kwargs.
        """
        if method == "tools/call":
            return self._route_by_name(params, "name", "tool")
        if method == "prompts/get":
            return self._route_by_name(params, "name", "prompt")
        if method in ("resources/read", "resources/subscribe",
                      "resources/unsubscribe"):
            return self._route_by_uri(params)
        if method == "completion/complete":
            return self._route_completion(params)
        return {"code": jsonrpc.METHOD_NOT_FOUND,
                "message": "method %r is not proxied by this gateway" % method}

    def _route_by_name(self, params, key, what):
        raw = params.get(key)
        if not isinstance(raw, str) or NAMESPACE_SEP not in raw:
            return {"code": jsonrpc.INVALID_PARAMS,
                    "message": "%s name %r is not namespaced as "
                               "<server>%sname" % (what, raw, NAMESPACE_SEP)}
        prefix, _, bare = raw.partition(NAMESPACE_SEP)
        up = self.upstreams.get(prefix)
        if up is None:
            return {"code": jsonrpc.INVALID_PARAMS,
                    "message": "no upstream named %r (known: %s)"
                               % (prefix, ", ".join(sorted(self.upstreams)))}
        rewritten = dict(params)
        rewritten[key] = bare
        return up, rewritten

    def _route_by_uri(self, params):
        uri = params.get("uri")
        if not isinstance(uri, str):
            return {"code": jsonrpc.INVALID_PARAMS,
                    "message": "uri must be a string"}
        # Resource URIs are not namespaced: a URI is meaningful to the server
        # that issued it and is shown to the user, so rewriting it would be a
        # visible lie. Routing uses the index built from resources/list.
        name = self._resource_owner.get(uri)
        if name is None:
            providers = [n for n, u in self.upstreams.items()
                         if "resources" in (u.capabilities or {})]
            if len(providers) == 1:
                name = providers[0]
            elif not providers:
                return {"code": jsonrpc.METHOD_NOT_FOUND,
                        "message": "no upstream provides resources"}
            else:
                return {"code": jsonrpc.INVALID_PARAMS,
                        "message": "cannot tell which upstream owns %r: it was "
                                   "not in any resources/list and %d upstreams "
                                   "provide resources" % (uri, len(providers))}
        return self.upstreams[name], dict(params)

    def _route_completion(self, params):
        ref = params.get("ref")
        if not isinstance(ref, dict):
            return {"code": jsonrpc.INVALID_PARAMS, "message": "ref must be an object"}
        if ref.get("type") == "ref/prompt":
            target = self._route_by_name(ref, "name", "prompt")
            if isinstance(target, dict):
                return target
            up, new_ref = target
            rewritten = dict(params)
            rewritten["ref"] = new_ref
            return up, rewritten
        if ref.get("type") == "ref/resource":
            target = self._route_by_uri(ref)
            if isinstance(target, dict):
                return target
            up, _ = target
            return up, dict(params)
        return {"code": jsonrpc.INVALID_PARAMS,
                "message": "unsupported completion ref type %r" % ref.get("type")}

    # ------------------------------------------------------ list aggregation

    def _handle_list(self, msg):
        method = msg["method"]
        key = {"tools/list": "tools", "prompts/list": "prompts",
               "resources/list": "resources",
               "resources/templates/list": "resourceTemplates"}[method]
        capability = {"tools": "tools", "prompts": "prompts",
                      "resources": "resources",
                      "resourceTemplates": "resources"}[key]

        # If nothing behind the gateway offers this primitive, answer the way
        # the upstreams would. Synthesising an empty list instead would be a
        # difference the client can observe, and "there are none" is a
        # materially different statement from "I do not implement that".
        if not any(capability in (up.capabilities or {})
                   for up in self.upstreams.values()):
            self._client.write(jsonrpc.error_response(
                msg["id"], jsonrpc.METHOD_NOT_FOUND,
                "no upstream provides %s" % capability))
            return

        results = self._fan_out(
            lambda up: (self._list_all(up, method, key)
                        if capability in (up.capabilities or {}) else []),
            timeout=self.config.request_timeout)

        merged = []
        errors = []
        for name in sorted(results):
            ok, value = results[name]
            if not ok:
                errors.append("%s: %s" % (name, value))
                continue
            for item in value:
                merged.append(self._namespace_item(name, key, item))

        if errors:
            # A partial list is a silent loss of tools the user configured.
            # Better to fail the call and say which upstream broke.
            self._client.write(jsonrpc.error_response(
                msg["id"], jsonrpc.INTERNAL_ERROR,
                "%s failed for %d upstream(s) -- %s"
                % (method, len(errors), "; ".join(errors))))
            return

        # Pagination is resolved here rather than exposed: every upstream page
        # is walked and the merged list returned whole, so there is no cursor
        # for the client to hold that would have to mean different offsets in
        # several upstreams at once.
        self._client.write(jsonrpc.result_response(msg["id"], {key: merged}))

    def _list_all(self, up, method, key):
        items = []
        cursor = None
        for _ in range(MAX_LIST_PAGES):
            params = {"cursor": cursor} if cursor else {}
            result = up.request(method, params, timeout=self.config.request_timeout) or {}
            page = result.get(key) or []
            if not isinstance(page, list):
                raise UpstreamError("%s returned a non-list %r" % (up.name, key))
            items.extend(page)
            cursor = result.get("nextCursor")
            if not cursor:
                return items
        raise UpstreamError("%s kept returning a nextCursor for %s after %d pages"
                            % (up.name, method, MAX_LIST_PAGES))

    def _namespace_item(self, name, key, item):
        if not isinstance(item, dict):
            return item
        # Copy rather than mutate: the only field that changes is the one the
        # gateway owns. Everything else, including fields this version has
        # never heard of, survives untouched.
        out = dict(item)
        if key in ("tools", "prompts") and isinstance(out.get("name"), str):
            out["name"] = "%s%s%s" % (name, NAMESPACE_SEP, out["name"])
            if len(out["name"]) > 128:
                log.warn("namespaced name %r is %d characters; some clients "
                         "reject names over 128" % (out["name"], len(out["name"])))
        if key == "resources" and isinstance(out.get("uri"), str):
            self._resource_owner[out["uri"]] = name
        return out

    def _handle_set_level(self, msg):
        params = msg.get("params") or {}
        results = self._fan_out(
            lambda up: (up.request("logging/setLevel", params,
                                   timeout=self.config.request_timeout)
                        if "logging" in (up.capabilities or {}) else None),
            timeout=self.config.request_timeout)
        errors = ["%s: %s" % (n, v) for n, (ok, v) in sorted(results.items()) if not ok]
        if errors:
            self._client.write(jsonrpc.error_response(
                msg["id"], jsonrpc.INTERNAL_ERROR,
                "logging/setLevel failed -- %s" % "; ".join(errors)))
            return
        self._client.write(jsonrpc.result_response(msg["id"], {}))

    def _fan_out(self, fn, timeout=None):
        """Run fn against every upstream concurrently.

        Returns {name: (ok, value_or_error_string)}. Never raises: a slow or
        broken upstream must not take the others down with it.

        A plain thread per upstream, not the handler pool. _fan_out is always
        called from a handler that is *itself* running on that pool, and a
        pool whose tasks wait on tasks submitted to the same pool deadlocks
        as soon as enough handlers are in flight to fill it -- four
        concurrent tools/list calls would have been enough to hang the
        gateway permanently.
        """
        out = {}
        out_lock = threading.Lock()

        def run(name, up):
            try:
                value = (True, fn(up))
            except (UpstreamError, UpstreamRequestError) as exc:
                value = (False, str(exc))
            except Exception as exc:  # noqa: BLE001
                value = (False, repr(exc))
            with out_lock:
                out[name] = value

        threads = []
        for name, up in self.upstreams.items():
            thread = threading.Thread(target=run, args=(name, up),
                                      name="gw-fan-%s" % name, daemon=True)
            thread.start()
            threads.append((name, thread))

        # One shared deadline, not one per thread: joining N threads for
        # `timeout` each would let a fan-out over five upstreams take five
        # times as long as the timeout the caller asked for.
        deadline = None if timeout is None else time.monotonic() + timeout
        for name, thread in threads:
            thread.join(None if deadline is None
                        else max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                with out_lock:
                    out.setdefault(name, (False, "timed out after %ss" % timeout))
        return out

    # ------------------------------------------------------ id bookkeeping

    def _forward_client_request(self, upstream, msg, params):
        gw_id = "c-%d" % next(self._ids)
        client_id = msg["id"]
        with self._lock:
            self._from_client[gw_id] = (upstream.name, client_id, msg.get("method"))
            self._client_id_index[_key(client_id)] = gw_id

        forwarded = dict(msg)
        forwarded["id"] = gw_id
        if params is not None:
            forwarded["params"] = params
        if not upstream.send(forwarded):
            with self._lock:
                self._from_client.pop(gw_id, None)
                self._client_id_index.pop(_key(client_id), None)
            self._client.write(jsonrpc.error_response(
                client_id, jsonrpc.INTERNAL_ERROR,
                "upstream %r is not available" % upstream.name))

    def _forward_client_cancel(self, params):
        requested = params.get("requestId")
        with self._lock:
            gw_id = self._client_id_index.get(_key(requested))
            entry = self._from_client.get(gw_id) if gw_id else None
        if entry is None:
            log.debug("cancellation for unknown request %r; dropped" % requested)
            return
        name = entry[0]
        rewritten = dict(params)
        rewritten["requestId"] = gw_id
        self.upstreams[name].notify("notifications/cancelled", rewritten)

    def _forward_client_response(self, msg):
        """A response to a request an upstream made of the client."""
        with self._lock:
            entry = self._from_upstream.pop(msg.get("id"), None)
            if entry:
                self._upstream_id_index.pop((entry[0], _key(entry[1])), None)
        if entry is None:
            log.debug("client responded to unknown request %r; dropped"
                      % msg.get("id"))
            return
        name, original_id = entry
        up = self.upstreams.get(name)
        if up is None:
            return
        restored = dict(msg)
        restored["id"] = original_id
        up.send(restored)

    # ----------------------------------------------------- upstream inbound

    def _on_upstream_message(self, up, msg):
        if jsonrpc.is_response(msg):
            with self._lock:
                entry = self._from_client.pop(msg.get("id"), None)
                if entry:
                    self._client_id_index.pop(_key(entry[1]), None)
            if entry is None:
                log.debug("%s: response to unknown request %r; dropped"
                          % (up.name, msg.get("id")))
                return
            _, client_id, _method = entry
            restored = dict(msg)
            restored["id"] = client_id
            # M1 hooks in here: this is where a tools/call result is
            # sanitized before it reaches the client and, through it, the model.
            self._client.write(restored)
            return

        if jsonrpc.is_request(msg):
            # sampling/createMessage, roots/list, elicitation/create.
            # Forwarded raw in M0; sampling carries content and is one of the
            # five PII paths the plan enumerates.
            gw_id = "u-%d" % next(self._ids)
            original = msg["id"]
            with self._lock:
                self._from_upstream[gw_id] = (up.name, original)
                self._upstream_id_index[(up.name, _key(original))] = gw_id
            forwarded = dict(msg)
            forwarded["id"] = gw_id
            self._client.write(forwarded)
            return

        if jsonrpc.is_notification(msg):
            self._on_upstream_notification(up, msg)

    def _on_upstream_notification(self, up, msg):
        method = msg.get("method")
        params = msg.get("params") or {}

        if method == "notifications/cancelled":
            with self._lock:
                gw_id = self._upstream_id_index.get(
                    (up.name, _key(params.get("requestId"))))
            if gw_id is None:
                log.debug("%s: cancellation for unknown request; dropped" % up.name)
                return
            rewritten = dict(msg)
            rewritten["params"] = dict(params, requestId=gw_id)
            self._client.write(rewritten)
            return

        if method in ("notifications/tools/list_changed",
                      "notifications/prompts/list_changed",
                      "notifications/resources/list_changed",
                      "notifications/resources/updated",
                      "notifications/message",
                      "notifications/progress"):
            self._client.write(msg)
            return

        log.debug("%s: notification %r not handled; dropped" % (up.name, method))

    def _on_upstream_exit(self, up):
        """Fail every call still waiting on an upstream that has died."""
        with self._lock:
            dead = [(gw_id, entry) for gw_id, entry in self._from_client.items()
                    if entry[0] == up.name]
            for gw_id, entry in dead:
                self._from_client.pop(gw_id, None)
                self._client_id_index.pop(_key(entry[1]), None)
        for _gw_id, (_name, client_id, method) in dead:
            self._client.write(jsonrpc.error_response(
                client_id, jsonrpc.INTERNAL_ERROR,
                "upstream %r exited while handling %r" % (up.name, method)))


def _key(value):
    """A hashable, type-aware key for a JSON-RPC id.

    JSON-RPC allows both numbers and strings, and 1 must not collide with
    "1" -- a client using string ids and one using integers would otherwise
    alias onto the same slot.
    """
    return (type(value).__name__, value)


def _version():
    from . import __version__
    return __version__
