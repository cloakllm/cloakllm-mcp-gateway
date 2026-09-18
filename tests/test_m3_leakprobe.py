"""M3: the release gate. Plant PII everywhere, assert none of it gets out.

This is the methodology from the held framework-integrations branch, where
two rounds of it found about nine real leaks. It transfers directly, and
the shape of every one of those leaks transfers with it: a reachable string
field the typed layer had not enumerated.

So the probe does not test the fields the walker knows about. It tests the
ones nobody thought of -- content types invented for this file, fields
nested inside fields, values in dictionary KEYS, PII in places the MCP spec
does not have today and might tomorrow.

Two rules, learned the hard way elsewhere in this project:

* Search for the planted value verbatim AND in digits-only form, so a
  reformatted copy cannot hide.
* Prove the probe can fail. Every invariant here has a companion that runs
  the identical session with sanitization off and asserts the leak IS
  found. A probe that cannot go red certifies nothing.
"""

import json

import pytest

from cloakllm_gateway import sanitizer, tripwire

# Deliberately all high-confidence regex categories: the probe is about
# reachability of surfaces, not about detection quality, and a NER miss
# would make it flaky for the wrong reason.
PLANTED = {
    "email": "marie.dubois@example-eu.fr",
    "card": "5500 0000 0000 0004",
    "iban": "FR76 3000 6000 0112 3456 7890 189",
    "aws_key": "AKIAIOSFODNN7EXAMPLE",
    "ssn": "123-45-6789",
}


def _leaked(blob):
    """Which planted values survive in this text, verbatim or as digits."""
    digits_only = "".join(c for c in blob if c.isdigit())
    out = []
    for label, value in PLANTED.items():
        if value in blob:
            out.append(label)
            continue
        digits = "".join(c for c in value if c.isdigit())
        if len(digits) >= 8 and digits in digits_only:
            out.append("%s (digits)" % label)
    return out


def _every_surface():
    """A result payload with PII on every surface we can think of.

    Including several the MCP spec does not have. That is the point: the
    walker must be right about payloads nobody wrote it against.
    """
    return {
        "content": [
            {"type": "text", "text": "plain %(email)s" % PLANTED},
            # A content type invented here. A closed allow-list would let
            # this through raw -- that was the LlamaIndex E1 BLOCKER.
            {"type": "some_future_type", "payload": PLANTED["card"]},
            {"type": "resource", "resource": {
                "uri": "file:///home/%(email)s/x.txt" % PLANTED,
                "mimeType": "text/plain",
                "text": "nested resource %(iban)s" % PLANTED}},
            {"type": "resource_link", "uri": "https://x/%(email)s" % PLANTED},
            # Binary is deliberately NOT scanned; see the limit asserted below.
            {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
        ],
        "structuredContent": {
            "rows": [{"customer": PLANTED["email"], "pan": PLANTED["card"]}],
            "deeply": {"nested": {"beyond": {"reason": PLANTED["ssn"]}}},
        },
        "isError": False,
        "_meta": {"note": "meta can carry %(aws_key)s too" % PLANTED},
        "x-vendor-extension": [[[PLANTED["iban"]]]],
        # A dictionary KEYED by an address. Only the tripwire covers this.
        "byCustomer": {PLANTED["email"]: {"total": 3}},
    }


# ------------------------------------------------------------ the invariant

def test_INVARIANT_no_planted_value_survives_any_surface():
    session = sanitizer.Session(sanitizer.build_shield({}))
    out, _ = session.sanitize_payload(_every_surface())
    out, _ = tripwire.OutboundTripwire(session).scrub(out)

    leaks = _leaked(json.dumps(out))
    # The image blob is excluded by design and carries no planted value, so
    # a clean result here is a real result.
    assert leaks == [], "leaked: %s" % leaks


def test_INVARIANT_the_probe_would_catch_a_leak():
    # Without this, the test above proves only that the payload builder and
    # the search function agree about nothing.
    leaks = _leaked(json.dumps(_every_surface()))
    assert set(PLANTED) <= {x.split(" ")[0] for x in leaks}, \
        "the probe cannot see planted values even with no sanitizer at all"


@pytest.mark.parametrize("label", sorted(PLANTED))
def test_each_planted_category_is_individually_caught(label):
    # A single blob assertion can pass because one category dominates.
    session = sanitizer.Session(sanitizer.build_shield({}))
    out, _ = session.sanitize_payload({"text": "value %s here" % PLANTED[label]})
    assert PLANTED[label] not in json.dumps(out)


def test_pii_hidden_in_a_dictionary_key_is_caught():
    # Found by this probe. The typed walker rewrites values only, because
    # renaming a protocol key would change the shape the client parses --
    # but a result can legitimately be a dictionary KEYED by an address,
    # and nothing else was covering that. The tripwire does it, safely,
    # because its regex gate never fires on "content" or "isError".
    session = sanitizer.Session(sanitizer.build_shield({}))
    payload = {PLANTED["email"]: "some value", "isError": False}

    typed, _ = session.sanitize_payload(payload)
    assert PLANTED["email"] in json.dumps(typed), "the typed pass skips keys"

    caught, hits = tripwire.OutboundTripwire(session).scrub(typed)
    assert PLANTED["email"] not in json.dumps(caught)
    assert "isError" in caught, "a structural key must keep its name"
    assert hits == 1


# ----------------------------------------------------------------- tripwire

def test_the_tripwire_catches_what_the_typed_walker_skips():
    # Simulate the exact failure the tripwire exists for: a surface the
    # typed pass left alone. "mimeType" is on the skip list, so PII there
    # survives the walker and must be caught on the second pass.
    session = sanitizer.Session(sanitizer.build_shield({}))
    payload = {"mimeType": "text/plain; note=%s" % PLANTED["email"]}

    typed, _ = session.sanitize_payload(payload)
    assert PLANTED["email"] in json.dumps(typed), "expected the typed pass to skip this"

    caught, hits = tripwire.OutboundTripwire(session).scrub(typed)
    assert hits == 1
    assert PLANTED["email"] not in json.dumps(caught)


def test_the_tripwire_leaves_clean_structural_strings_alone():
    # Regex-only is what makes it safe to run on everything: no NER, so no
    # name or organisation false positives on ids, mime types or URLs.
    session = sanitizer.Session(sanitizer.build_shield({}))
    payload = {"type": "text", "mimeType": "application/json",
               "uri": "https://example.com/a/b?c=1",
               "text": "Marie Dubois of Acme Corp in France"}
    out, hits = tripwire.OutboundTripwire(session).scrub(payload)

    assert hits == 0
    assert out == payload


def test_the_tripwire_fails_open():
    # It is a safety net, never a new way to drop traffic. If it breaks,
    # the typed result -- which already ran -- goes out unchanged.
    session = sanitizer.Session(sanitizer.build_shield({}))
    wire = tripwire.OutboundTripwire(session)
    wire._has_pii = lambda _t: (_ for _ in ()).throw(RuntimeError("boom"))

    payload = {"text": "anything"}
    out, hits = wire.scrub(payload)
    assert out == payload and hits == 0


def test_the_tripwire_never_logs_the_value_it_found():
    # No-PII-in-logs applies to warnings too, and a tripwire warning is
    # written at exactly the moment PII is in hand.
    import io
    import sys
    session = sanitizer.Session(sanitizer.build_shield({}))
    from cloakllm_gateway import log as gwlog
    gwlog.set_level("debug")
    captured, sys.stderr = io.StringIO(), io.StringIO()
    real, sys.stderr = sys.stderr, captured
    try:
        tripwire.OutboundTripwire(session).scrub(
            {"mimeType": "x; note=%s" % PLANTED["email"]})
    finally:
        sys.stderr = real
        gwlog.set_level("silent")

    assert _leaked(captured.getvalue()) == []
    assert "TRIPWIRE" in captured.getvalue()


def test_binary_content_is_not_scanned_and_that_is_documented():
    # An explicit limit, not an oversight. Regex cannot find a card number
    # inside a JPEG -- the bytes are not there as text -- while a chance
    # digit run in the base64 alphabet would corrupt the image for nothing.
    # A screenshot of a card passes through this gateway. The README says so.
    assert "data" in tripwire.BINARY_KEYS and "blob" in tripwire.BINARY_KEYS


# ------------------------------------------- cross-upstream token provenance

def test_a_token_from_one_upstream_is_flagged_when_sent_to_another(harness):
    # The exfiltration path the round trip creates. `fs` returns a customer
    # record, the model sees [EMAIL_0], and the model then calls `db` with
    # [EMAIL_0]. The gateway restores it, so a value the model was never
    # allowed to see reaches a server that was never meant to have it --
    # with the gateway doing the handing over.
    session = sanitizer.Session(sanitizer.build_shield({}))
    session.sanitize_payload({"text": "customer %(email)s" % PLANTED}, origin="fs")

    foreign = session.tokens_from_other_upstreams({"to": "[EMAIL_0]"}, "db")
    assert foreign == ["EMAIL_0"]
    assert session.tokens_from_other_upstreams({"to": "[EMAIL_0]"}, "fs") == []


def test_token_scope_upstream_refuses_the_cross_call(harness):
    h = harness({"fs": {}, "db": {}}, sanitize=True, token_scope="upstream")
    h.initialize()
    seen = _result(h, h.request("tools/call", {
        "name": "fs__echo", "arguments": {"r": PLANTED["email"]}}))
    assert "[EMAIL_0]" in seen["content"][0]["text"]

    refused = h.request("tools/call", {
        "name": "db__echo", "arguments": {"to": "[EMAIL_0]"}})
    assert "error" in refused
    assert "different upstream" in refused["error"]["message"]


def test_token_scope_session_allows_it_but_says_so(harness):
    # The default. Moving data between tools is a real workflow and the
    # reason to run several servers behind one gateway at all, so it is
    # allowed -- visibly.
    h = harness({"fs": {}, "db": {}}, sanitize=True)
    h.initialize()
    _result(h, h.request("tools/call", {
        "name": "fs__echo", "arguments": {"r": PLANTED["email"]}}))

    allowed = h.request("tools/call", {
        "name": "db__echo", "arguments": {"to": "[EMAIL_0]"}})
    assert "result" in allowed


def test_the_same_upstream_is_never_flagged(harness):
    # The check has to be quiet on the overwhelmingly common case or it is
    # just another warning people learn to scroll past.
    h = harness({"fs": {}}, sanitize=True, token_scope="upstream")
    h.initialize()
    _result(h, h.request("tools/call", {
        "name": "fs__echo", "arguments": {"r": PLANTED["email"]}}))
    again = h.request("tools/call", {
        "name": "fs__echo", "arguments": {"to": "[EMAIL_0]"}})

    assert "result" in again


def _result(_h, response):
    assert "error" not in response, response["error"]
    return response["result"]
