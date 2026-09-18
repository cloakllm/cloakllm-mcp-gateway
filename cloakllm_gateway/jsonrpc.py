"""JSON-RPC 2.0 message framing for the MCP stdio transport.

Two deliberate choices here, both about fidelity:

1. **Messages are plain dicts, never typed models.** A proxy that parses
   incoming messages into a typed schema silently drops every field the schema
   does not know about, and the MCP spec grows new fields constantly. Dropping
   an unknown field is a correctness bug the client cannot see. So the gateway
   manipulates the raw decoded object and re-serialises it, touching only the
   keys it has a reason to touch.

2. **Binary streams, explicit UTF-8, explicit "\\n".** Python's text mode on
   Windows translates "\\n" to "\\r\\n" on write, which corrupts newline-delimited
   framing for strict parsers, and decodes with the console codepage on read
   (cp1255 on this project's dev machine), which mangles any non-ASCII payload.
   Both failures are invisible until someone runs the gateway on Windows.
"""

import json
import threading

# A hostile or broken upstream can emit an unbounded line. readline() buffers
# the whole thing before we get a chance to look at it, so this cannot prevent
# the allocation -- it exists so that one oversized message is reported and
# skipped instead of being parsed and forwarded.
DEFAULT_MAX_MESSAGE_BYTES = 32 * 1024 * 1024

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class MessageTooLarge(Exception):
    """A single JSON-RPC line exceeded the configured limit."""


class MalformedMessage(Exception):
    """A line could not be decoded as a JSON-RPC message."""


def read_message(stream, max_bytes=DEFAULT_MAX_MESSAGE_BYTES):
    """Read one newline-delimited JSON message from a binary stream.

    Returns the decoded object, or None at end of stream. Blank lines are
    skipped: some servers emit them between messages and the spec does not
    forbid it.
    """
    while True:
        line = stream.readline()
        if not line:
            return None
        if len(line) > max_bytes:
            raise MessageTooLarge(
                "message of %d bytes exceeds the %d byte limit"
                % (len(line), max_bytes))
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise MalformedMessage(str(exc)) from exc
        if not isinstance(obj, dict):
            # Batches (a JSON array) were removed in MCP 2025-06-18 and were
            # never widely used. Refusing is honest; guessing is not.
            raise MalformedMessage(
                "expected a JSON-RPC object, got %s" % type(obj).__name__)
        return obj


def encode_message(obj):
    """Serialise one message to the bytes that go on the wire.

    json.dumps escapes newlines inside strings whatever ensure_ascii is set
    to, so the framing is safe with ensure_ascii=False and the payload stays
    the size the spec expects rather than being inflated by \\uXXXX escapes.
    """
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"


class MessageWriter:
    """Serialises writes to one binary stream.

    Several threads write to the same pipe -- the client reader, every
    upstream reader, and the gateway itself -- so interleaving two messages
    would produce one unparseable line.
    """

    def __init__(self, stream):
        self._stream = stream
        self._lock = threading.Lock()
        self._closed = False

    def write(self, obj):
        """Write one message. Returns False if the stream is already gone."""
        data = encode_message(obj)
        with self._lock:
            if self._closed:
                return False
            try:
                self._stream.write(data)
                self._stream.flush()
                return True
            except (OSError, ValueError):
                # The peer went away mid-write. Callers treat this as a
                # disconnect, not as an error to report back to anyone.
                self._closed = True
                return False

    def close(self):
        with self._lock:
            self._closed = True
            try:
                self._stream.close()
            except (OSError, ValueError):
                pass


def is_request(msg):
    """A request has both a method and an id. A notification has no id."""
    return "method" in msg and "id" in msg and msg["id"] is not None


def is_notification(msg):
    return "method" in msg and ("id" not in msg or msg["id"] is None)


def is_response(msg):
    return "method" not in msg and "id" in msg


def error_response(msg_id, code, message, data=None):
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": msg_id, "error": err}


def result_response(msg_id, result):
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}
