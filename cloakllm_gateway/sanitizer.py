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
    "progressToken",  # correlates a progress notification with its request
})
BINARY_KEYS = frozenset({"data", "blob"})
SKIP_KEYS = IDENTITY_KEYS | BINARY_KEYS

# uri and uriTemplate were in the skip set at M1, because tokenizing a URI
# needs a return path to put it back and there was none. M2 added one, so
# they are sanitized now: a URI is shown to the model like any other text
# and "file:///home/marie.dubois@example-eu.fr/notes.txt" is exactly the
# kind of thing this product exists to keep out of a provider's context.
# Routing still works because the gateway desanitizes an inbound request
# BEFORE it routes on it, and the resource index holds the real URIs.

# The response methods whose payloads reach the model's context.
#
# resources/list is here and tools/list is not, which looks inconsistent
# until you look at what they carry. tools/list is schemas -- rewriting a
# description or an inputSchema enum corrupts the contract the model calls
# against, for no privacy gain. resources/list is a listing of real things:
# on a filesystem server it is a directory listing, and filenames carry
# names and addresses.
SANITIZED_METHODS = frozenset({
    "tools/call", "resources/read", "prompts/get",
    "resources/list", "resources/templates/list",
})


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

    def __init__(self, shield, token_scope="session"):
        self._shield = shield
        self._token_map = None
        # Shield.sanitize mutates the shared map, and tool results arrive on
        # one reader thread per upstream. Without this, two concurrent calls
        # can both allocate [EMAIL_0] for different addresses.
        self._lock = threading.Lock()
        self.token_scope = token_scope
        # Which upstream's data each token stands for. See _check_provenance
        # for what this is actually defending against.
        self._origin = {}

    @property
    def shield(self):
        return self._shield

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

    def sanitize_payload(self, payload, origin=None):
        """Return a sanitized copy of an arbitrary JSON payload.

        Raises SanitizationFailed if any part of it could not be handled, so
        the caller can refuse the whole message rather than forward some of
        it unprotected.

        `origin` is the upstream whose data this is, recorded per token so
        that a later request can be checked against it.
        """
        out, stats = self._transform(payload, self.sanitize_text, "sanitize")
        if origin is not None:
            for token in _tokens_in(out):
                self._origin.setdefault(token, origin)
        return out, stats

    def origin_of(self, token):
        """Which upstream's data a token stands for, or None if unknown."""
        return self._origin.get(token)

    def tokens_from_other_upstreams(self, payload, target):
        """Tokens in an outbound payload that stand for another server's data.

        This is the exfiltration path the round trip creates, and it is not
        obvious. Suppose `fs` returns a customer record and the model sees
        [EMAIL_0]. The model then calls a third-party search or
        issue-tracker server with [EMAIL_0] in the arguments. The gateway
        dutifully restores it, and the address the model was never allowed
        to see is handed to a party that was never meant to have it -- with
        the gateway itself performing the exfiltration.

        Before the gateway, the model could not have done this: it never
        had the value. Tokenization is what made the value *portable*
        without being *visible*. That is a new capability and it needs to
        be visible to the operator.
        """
        return sorted({t for t in _tokens_in(payload)
                       if self._origin.get(t) not in (None, target)})

    def desanitize_payload(self, payload):
        """Return a copy with tokens restored to the values they stand for.

        This runs on the way OUT, towards a real MCP server, and it is what
        makes the round trip work: the model reasons over [EMAIL_0] and the
        tool still receives the address.

        Note what this is and is not. Desanitizing here is a *functionality*
        guarantee, not a privacy one -- the protection already happened when
        the value was tokenized on the way in. The privacy question on this
        leg is a different one, and _detect_outbound below is what asks it.
        """
        return self._transform(payload, self.desanitize_text, "desanitize")

    def _transform(self, payload, fn, what):
        stats = Stats()
        try:
            out = self._walk(payload, None, stats, fn)
        except Exception as exc:  # noqa: BLE001 - fail closed on anything
            raise SanitizationFailed("%s: %s" % (what, exc)) from exc
        if stats.skipped:
            log.debug("%s left %s untouched (identity or binary fields)"
                      % (what, ", ".join("%s x%d" % (k, n)
                                         for k, n in sorted(stats.skipped.items()))))
        return out, stats

    def _walk(self, node, key, stats, fn):
        if isinstance(node, str):
            stats.strings += 1
            if key in SKIP_KEYS:
                stats.note_skip(key)
                return node
            stats.sanitized += 1
            out = fn(node)
            if out != node:
                stats.changed += 1
            return out
        if isinstance(node, dict):
            # A copy, never a mutation: the caller may still hold the
            # original, and at M4 the audit hook will want the pre-image.
            return {k: self._walk(v, k, stats, fn) for k, v in node.items()}
        if isinstance(node, list):
            # Carry the parent key down so a list of strings under "data"
            # is still recognised as binary.
            return [self._walk(v, key, stats, fn) for v in node]
        # bool / int / float / None carry no text.
        return node

    def detect_outbound(self, payload):
        """Report PII in an outbound payload that is NOT one of our tokens.

        A token being restored here is the system working. A raw address the
        model wrote out itself is something else, and worth surfacing:
        upstreams are not all equal. A filesystem server on the same machine
        receiving a real customer record is fine; a third-party search or
        issue-tracker server receiving the same record is an exfiltration
        path that happens to be shaped like a tool call, and the gateway has
        no way to tell the two apart from the config alone.

        Detection only -- nothing is blocked or rewritten. Per-upstream
        trust, which is what would let this become enforcement, is a later
        decision and needs the operator to declare it.
        """
        found = {}
        for text in _strings(payload):
            # A throwaway map on purpose: this is a probe, and it must not
            # allocate tokens in the session the model is actually using.
            with self._lock:
                _, probe_map = self._shield.sanitize(text)
            for category in _categories(probe_map):
                found[category] = found.get(category, 0) + 1
        return found


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


def tokens_in(node):
    """Every CloakLLM token appearing anywhere in a payload."""
    found = set()
    for text in _strings(node):
        found.update(_TOKEN_RE.findall(text))
    return found


_tokens_in = tokens_in


def _token_regex():
    # Prefer the SDK's canonical pattern so the two cannot drift; fall back
    # to the documented [CATEGORY_N] shape if that module ever moves.
    try:
        from cloakllm.token_spec import CLOAKLLM_TOKEN_REGEX
        return CLOAKLLM_TOKEN_REGEX
    except Exception:  # pragma: no cover - the SDK always ships it today
        import re
        return re.compile(r"\[[A-Z][A-Z0-9_]*_\d+\]")


_TOKEN_RE = _token_regex()


def _strings(node, key=None):
    """Yield every string in a payload that the walker would have touched."""
    if isinstance(node, str):
        if key not in SKIP_KEYS:
            yield node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _strings(v, k)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v, key)


def _categories(token_map):
    """The category names a TokenMap found, however the SDK spells them."""
    categories = getattr(token_map, "categories", None) or []
    if isinstance(categories, dict):
        return list(categories)
    return list(categories)


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
