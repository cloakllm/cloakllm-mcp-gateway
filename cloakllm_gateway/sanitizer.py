"""Sanitizing MCP payloads on their way to the model.

## Why the walker sanitizes by default and skips by exception

The LlamaIndex integration shipped a closed allow-list of content-block
types, and an audit found the BLOCKER: a type the list had never heard of
went through raw. MCP content types will keep growing -- text, image, audio,
resource links, whatever ships next -- so the default here is the opposite.
Every string anywhere in a payload is sanitized unless it falls in a small,
principled set of exceptions, and each exception is *counted* rather than
silently dropped, so M3's tripwire has an explicit list of surfaces to scan.

The exceptions are all identity-bearing or binary: values whose exact bytes
have to survive for the protocol to keep working.

## Why the token map is per SESSION, not per call

The framework integrations used a per-call map, correctly: there the map's
whole lifetime was one request. Here the opposite is true, and a per-call
map would break the product. The model sees `[EMAIL_0]` in the result of
call 1 and may pass it back as an argument to call 5; if the map died with
call 1, there would be nothing left to resolve it against. The map lives as
long as the client's session and dies with it.
"""

import threading

from . import log

# Values whose exact bytes must survive. Every one of these is either a
# protocol discriminator, a routing identity, or base64 binary -- never free
# text. Anything NOT listed here gets sanitized, including fields added to
# MCP after this was written.
IDENTITY_KEYS = frozenset({
    "type",           # the content-block discriminator; rewriting it breaks parsing
    "mimeType",
    "uri",            # routing identity. Tokenizing these needs M2's return path
    "uriTemplate",
    "progressToken",  # correlates a progress notification with its request
})
BINARY_KEYS = frozenset({"data", "blob"})
SKIP_KEYS = IDENTITY_KEYS | BINARY_KEYS

# The response methods whose payloads reach the model's context.
SANITIZED_METHODS = frozenset({"tools/call", "resources/read", "prompts/get"})


class SanitizationFailed(Exception):
    """Sanitization could not complete, so nothing may be forwarded.

    Fail-closed is the deliberate choice: losing a tool call is bad, leaking
    PII is the thing this exists to prevent.
    """


class Stats:
    __slots__ = ("strings", "sanitized", "skipped", "changed")

    def __init__(self):
        self.strings = 0
        self.sanitized = 0
        self.skipped = {}     # key -> count, the surfaces M3 must scan
        self.changed = 0

    def note_skip(self, key):
        self.skipped[key] = self.skipped.get(key, 0) + 1

    def __repr__(self):
        return ("Stats(strings=%d sanitized=%d changed=%d skipped=%r)"
                % (self.strings, self.sanitized, self.changed, self.skipped))


class Session:
    """One client's sanitization state: a shield and a token map.

    With the stdio transport there is exactly one session per process, but
    the state is held here rather than on the gateway so that a multi-client
    transport cannot accidentally inherit a shared map. That cross-request
    bleed -- user A's PII surfacing in user B's response -- is a bug this
    project has already paid for once.
    """

    def __init__(self, shield):
        self._shield = shield
        self._token_map = None
        # Shield.sanitize mutates the shared map, and tool results arrive on
        # one reader thread per upstream. Without this, two concurrent calls
        # can both allocate [EMAIL_0] for different addresses.
        self._lock = threading.Lock()

    @property
    def token_map(self):
        return self._token_map

    def sanitize_text(self, text):
        with self._lock:
            out, self._token_map = self._shield.sanitize(
                text, token_map=self._token_map)
            return out

    def desanitize_text(self, text):
        """Restore real values. Used from M2 onward for tool arguments."""
        with self._lock:
            if self._token_map is None:
                return text
            return self._shield.desanitize(text, self._token_map)

    # ------------------------------------------------------------- walking

    def sanitize_payload(self, payload):
        """Return a sanitized copy of an arbitrary JSON payload.

        Raises SanitizationFailed if any part of it could not be handled, so
        the caller can refuse the whole message rather than forward some of
        it unprotected.
        """
        stats = Stats()
        try:
            out = self._walk(payload, None, stats)
        except Exception as exc:  # noqa: BLE001 - fail closed on anything
            raise SanitizationFailed(str(exc)) from exc
        if stats.skipped:
            log.debug("left %s untouched (identity or binary fields)"
                      % ", ".join("%s x%d" % (k, n)
                                  for k, n in sorted(stats.skipped.items())))
        return out, stats

    def _walk(self, node, key, stats):
        if isinstance(node, str):
            stats.strings += 1
            if key in SKIP_KEYS:
                stats.note_skip(key)
                return node
            stats.sanitized += 1
            out = self.sanitize_text(node)
            if out != node:
                stats.changed += 1
            return out
        if isinstance(node, dict):
            # A copy, never a mutation: the caller may still hold the
            # original, and at M4 the audit hook will want the pre-image.
            return {k: self._walk(v, k, stats) for k, v in node.items()}
        if isinstance(node, list):
            # Carry the parent key down so a list of strings under "data"
            # is still recognised as binary.
            return [self._walk(v, key, stats) for v in node]
        # bool / int / float / None carry no text.
        return node


# The SDK enables every NER label it knows: PERSON, ORG, GPE, LOC, NORP,
# FAC, MISC and the non-English equivalents. That is right for the SDK and
# wrong here.
#
# A gateway tokenizes what the model then has to reason over, so a false
# positive is not free: with the full set, "what is the capital of France"
# reaches the model as "the capital of [GPE_0]" and the answer is gone.
# GPE and ORG are also much weaker signals of personal data than a name --
# "France" and "Acme Corp" identify nobody. So the default is PERSON only,
# in the label spellings the SDK's supported models use. Operators who want
# the wider set can say so with ner_entity_types.
DEFAULT_NER_ENTITY_TYPES = frozenset({"PERSON", "PER", "persName", "PS"})

# Probes for the startup self-test, one per togglable category.
_SELF_TEST_PROBES = (
    ("detect_emails", "selftest.probe@example-eu.fr"),
    ("detect_credit_cards", "5500 0000 0000 0004"),
    ("detect_iban", "FR76 3000 6000 0112 3456 7890 189"),
    ("detect_api_keys", "AKIAIOSFODNN7EXAMPLE"),
    ("detect_ssns", "123-45-6789"),
)


def self_test(session, detection_options):
    """Prove the detector actually detects, before serving any traffic.

    Constructing a Shield is not evidence that it works. A spaCy model that
    failed to load, or a detection option that quietly did nothing, both
    produce a perfectly healthy-looking object that sanitizes nothing -- and
    the gateway would come up announcing "sanitization enabled" while
    forwarding everything in the clear. So run real values through the real
    path and check they came out changed.

    Only categories the operator left enabled are probed, so deliberately
    narrowing detection does not trip it.
    """
    options = detection_options or {}
    checked = []
    for flag, probe in _SELF_TEST_PROBES:
        if not options.get(flag, True):
            continue
        out = session.sanitize_text("self test value %s end" % probe)
        if probe in out:
            raise SanitizationFailed(
                "startup self-test failed: %s is enabled but a known %s "
                "value passed through unchanged. Refusing to start rather "
                "than report protection that is not happening."
                % (flag, flag.replace("detect_", "")))
        checked.append(flag)
    if not checked:
        log.warn("startup self-test skipped: every probed detection "
                 "category is disabled")
    else:
        log.info("startup self-test passed for %s" % ", ".join(checked))
    return checked


def build_shield(detection_options):
    """Construct the Shield the gateway sanitizes with.

    Imported lazily so that the gateway still starts, and reports a usable
    error, when cloakllm is not installed.
    """
    try:
        from cloakllm import Shield, ShieldConfig
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise SanitizationFailed(
            "sanitization is enabled but cloakllm is not installed: %s. "
            "Install it, or set \"sanitize\": false to run as a plain proxy."
            % exc) from exc

    options = dict(detection_options or {})
    options.setdefault("ner_entity_types", set(DEFAULT_NER_ENTITY_TYPES))
    # The gateway does not write the SDK's own audit chain. M4 adds a
    # gateway-shaped one that records tool calls; letting the SDK log every
    # individual string as a separate sanitize event would bury it.
    options["audit_enabled"] = False
    return Shield(ShieldConfig(**options))
