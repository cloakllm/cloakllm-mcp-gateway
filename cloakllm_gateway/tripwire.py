"""The safety net for the surface nobody thought of.

The typed walker in sanitizer.py is deny-by-default, which is the right
shape, but it can still only be right about the payloads it was written
against. MCP grows content types; an upstream can return anything JSON can
express. Every leak found in the two audit rounds on the held LangChain /
LlamaIndex branch was the same shape -- a reachable string field the typed
layer had not enumerated -- and there is no reason this proxy is immune.

So after the typed pass, this walks the ENTIRE payload again and, for every
string, runs a regex-ONLY detection pass. If deterministic high-confidence
PII survived, it scrubs that string with the same token map before the
payload reaches the client and, through it, the model.

Three properties make it safe to run on every single message:

* **Regex only.** No NER, no LLM pass. Only email / SSN / card / IBAN / IP /
  phone / key shapes are touched, so structural strings -- ids, mime types,
  discriminators -- are never altered and there are no name or organisation
  false positives.
* **Scrub on hit, and warn once.** A hit means the typed layer missed a
  field. The tripwire stops the leak AND says so, so the real gap gets
  found and closed rather than silently papered over.
* **Fail-open.** Any error leaves the payload exactly as the typed
  sanitizer produced it, and that already ran. The tripwire can only ever
  add protection, never remove it.

The warning never contains the value it found. No-PII-in-logs applies to
warnings too.
"""

from . import log

# Base64 payloads are excluded, and this is a real limit rather than an
# oversight. Regex on base64 cannot find a card number inside a JPEG -- the
# bytes simply are not there as text -- while a chance 16-digit run in the
# base64 alphabet would corrupt the image for nothing. Scanning it would
# buy false positives and no protection. The README says so plainly:
# image and audio content is not inspected.
BINARY_KEYS = frozenset({"data", "blob"})


class OutboundTripwire:
    def __init__(self, session):
        self._session = session
        self._warned = False
        self._hits = 0
        self._regex = None
        try:
            from cloakllm.backends.regex import RegexBackend
            self._regex = RegexBackend(session.shield.config)
        except Exception as exc:  # noqa: BLE001 - the net must not be the hole
            log.warn("tripwire disabled: could not build the regex backend "
                     "(%r). The typed sanitizer still runs." % exc)

    @property
    def hits(self):
        return self._hits

    def _has_pii(self, text):
        if not self._regex or not text:
            return False
        try:
            return bool(self._regex.detect(text, []))
        except Exception:  # noqa: BLE001 - detection must never break a call
            return False

    def scrub(self, payload):
        """Return (payload, hit_count). Never raises."""
        if self._regex is None:
            return payload, 0
        try:
            out, hits = self._walk(payload, None)
        except Exception as exc:  # noqa: BLE001 - fail OPEN, deliberately
            log.debug("tripwire walk failed, leaving the typed result as-is: %r" % exc)
            return payload, 0
        if hits:
            self._hits += hits
            if not self._warned:
                self._warned = True
                log.warn("TRIPWIRE: high-confidence PII survived the typed "
                         "sanitizer on %d string(s) and was scrubbed. This is "
                         "a gap in the typed walker -- please report the "
                         "shape of the payload (not its contents)." % hits)
        return out, hits

    def _walk(self, node, key):
        if isinstance(node, str):
            if key in BINARY_KEYS or not self._has_pii(node):
                return node, 0
            return self._session.sanitize_text(node), 1
        if isinstance(node, dict):
            out = {}
            hits = 0
            for k, v in node.items():
                # Keys too. The typed walker rewrites values only, because
                # renaming a protocol key would change the shape the client
                # parses -- but a result can legitimately be a dictionary
                # KEYED by an address, and that is a real leak with no
                # other net under it. The regex-only gate is what makes
                # this safe: "content", "type" and "isError" contain no
                # PII and are never touched.
                new_key = k
                if isinstance(k, str) and k not in BINARY_KEYS and self._has_pii(k):
                    new_key = self._session.sanitize_text(k)
                    hits += 1
                out[new_key], n = self._walk(v, k)
                hits += n
            return out, hits
        if isinstance(node, list):
            out = []
            hits = 0
            for v in node:
                item, n = self._walk(v, key)
                out.append(item)
                hits += n
            return out, hits
        return node, 0
