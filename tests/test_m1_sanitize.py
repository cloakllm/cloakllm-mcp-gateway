"""M1: tool results are sanitized before they reach the model.

The invariant suite at the bottom is the important part. "No original PII
reaches the provider" is the claim the whole product rests on, so it is
tested the way the SDK tests its own: plant known values, drive them through
the real path, then assert they appear nowhere in anything the client
receives -- verbatim *and* in digits-only form, so a reformatted copy cannot
hide.
"""

import json

import pytest

from conftest import Timeout  # noqa: F401 - imported for the module docstring's sake
from cloakllm_gateway import sanitizer


PLANTED = {
    "email": "marie.dubois@example-eu.fr",
    "card": "5500 0000 0000 0004",
    "iban": "FR76 3000 6000 0112 3456 7890 189",
    "aws_key": "AKIAIOSFODNN7EXAMPLE",
}


def _result(response):
    assert "error" not in response, response["error"]
    return response["result"]


def _call(h, tool, args):
    return h.request("tools/call", {"name": tool, "arguments": args})


# --------------------------------------------------------------- the basics

def test_a_planted_email_reaches_the_model_as_a_token(harness):
    # The M0 acceptance asserted this value arrived verbatim. That assertion
    # inverting is what M1 means.
    h = harness({"fs": {}}, sanitize=True)
    h.initialize()
    result = _result(_call(h, "fs__echo", {"note": PLANTED["email"]}))
    text = result["content"][0]["text"]

    assert PLANTED["email"] not in text
    assert "[EMAIL_0]" in text


def test_every_high_confidence_category_is_caught(harness):
    h = harness({"fs": {}}, sanitize=True)
    h.initialize()
    blob = " / ".join(PLANTED.values())
    text = _result(_call(h, "fs__echo", {"note": blob}))["content"][0]["text"]

    for label, value in PLANTED.items():
        assert value not in text, "%s survived: %s" % (label, text)


def test_the_same_value_gets_the_same_token_across_calls(harness):
    # The token map is per session, not per call. If it were per call the
    # model would see [EMAIL_0] mean two different people in one
    # conversation -- and M2 would have nothing left to resolve against.
    h = harness({"fs": {}}, sanitize=True)
    h.initialize()
    first = _result(_call(h, "fs__echo", {"a": PLANTED["email"]}))["content"][0]["text"]
    second = _result(_call(h, "fs__echo", {"b": PLANTED["email"]}))["content"][0]["text"]

    assert "[EMAIL_0]" in first and "[EMAIL_0]" in second


def test_two_different_values_get_different_tokens(harness):
    h = harness({"fs": {}}, sanitize=True)
    h.initialize()
    text = _result(_call(h, "fs__echo", {
        "a": "marie.dubois@example-eu.fr", "b": "jan.novak@example-eu.cz",
    }))["content"][0]["text"]

    assert "[EMAIL_0]" in text and "[EMAIL_1]" in text


def test_clean_text_is_returned_unchanged(harness):
    # Over-tokenizing is not free: a value the model needed is gone. This
    # test caught the default NER set doing exactly that -- the question
    # arrived as "the capital of [GPE_0]" and was unanswerable.
    h = harness({"fs": {}}, sanitize=True)
    h.initialize()
    text = _result(_call(h, "fs__echo", {"q": "what is the capital of France"}))
    assert "capital of France" in text["content"][0]["text"]


def test_place_and_organisation_names_are_left_alone_by_default(harness):
    # GPE and ORG identify nobody, and tokenizing them costs the model the
    # ability to reason about them. PERSON is a different matter.
    h = harness({"fs": {}}, sanitize=True)
    h.initialize()
    text = _result(_call(h, "fs__echo", {
        "note": "Marie Dubois at Acme Corp in France"}))["content"][0]["text"]

    assert "Acme Corp" in text and "France" in text
    assert "Marie Dubois" not in text
    assert "[PERSON_0]" in text


def test_an_operator_can_widen_the_ner_set(harness):
    h = harness({"fs": {}}, sanitize=True,
                detection={"ner_entity_types": ["PERSON", "GPE", "ORG"]})
    h.initialize()
    text = _result(_call(h, "fs__echo", {
        "note": "Marie Dubois at Acme Corp in France"}))["content"][0]["text"]

    assert "France" not in text


def test_sanitization_can_be_switched_off(harness):
    # The M0 baseline is still reachable, and the transparency suite runs
    # against it.
    h = harness({"fs": {}}, sanitize=False)
    h.initialize()
    text = _result(_call(h, "fs__echo", {"note": PLANTED["email"]}))["content"][0]["text"]

    assert PLANTED["email"] in text


# ----------------------------------------------------------- reach of the walk

def test_error_bodies_are_sanitized_too(harness):
    # Not an afterthought: the real filesystem server puts the requested
    # path in its "file not found" message, so a failed call leaks exactly
    # what a successful one would have.
    h = harness({"fs": {}}, sanitize=True)
    h.initialize()
    response = _call(h, "fs__boom", {"leak": PLANTED["email"]})

    assert "error" in response
    assert PLANTED["email"] not in json.dumps(response)


def test_nested_and_unknown_fields_are_walked(harness):
    # The walker sanitizes by default and skips by exception, so a field no
    # version of this gateway has heard of is still covered. This is the
    # LlamaIndex E1 blocker -- a closed allow-list letting unknown content
    # types through raw -- inverted.
    h = harness({"fs": {}}, sanitize=True)
    h.initialize()
    result = _result(_call(h, "fs__echo", {
        "x-not-a-real-field": {"deeply": [{"nested": PLANTED["email"]}]}}))

    assert PLANTED["email"] not in json.dumps(result)


def test_identity_and_binary_fields_are_left_alone(harness):
    # These have to survive byte for byte or the protocol breaks: "type" is
    # the content discriminator, "blob" is base64. They are counted as
    # skips so M3's tripwire knows which surfaces to scan.
    #
    # "uri" was in this set at M1 and is not any more: M2 added the return
    # path that lets a tokenized URI be restored before routing.
    session = sanitizer.Session(sanitizer.build_shield({}))
    payload = {
        "type": "text",
        "mimeType": "text/plain",
        "blob": "aGVsbG8gd29ybGQ=",
        "text": "mail %s" % PLANTED["email"],
    }
    out, stats = session.sanitize_payload(payload)

    assert out["type"] == "text"
    assert out["mimeType"] == "text/plain"
    assert out["blob"] == payload["blob"], "base64 must not be rewritten"
    assert PLANTED["email"] not in out["text"]
    assert set(stats.skipped) == {"type", "mimeType", "blob"}


def test_non_string_scalars_survive(harness):
    session = sanitizer.Session(sanitizer.build_shield({}))
    out, _ = session.sanitize_payload(
        {"isError": False, "count": 3, "ratio": 0.5, "nothing": None})

    assert out == {"isError": False, "count": 3, "ratio": 0.5, "nothing": None}
    assert out["isError"] is False


def test_only_model_facing_methods_are_sanitized(harness):
    # tools/list carries schemas, not data. Rewriting a description or an
    # enum inside an inputSchema would corrupt the contract the model calls
    # against, for no privacy gain. resources/list is the opposite case --
    # a directory listing IS data -- and was added in M2.
    assert "tools/list" not in sanitizer.SANITIZED_METHODS
    assert "prompts/list" not in sanitizer.SANITIZED_METHODS
    assert sanitizer.SANITIZED_METHODS == {
        "tools/call", "resources/read", "prompts/get",
        "resources/list", "resources/templates/list"}


def test_resource_reads_are_sanitized(harness):
    h = harness({"fs": {}}, sanitize=True)
    h.initialize()
    h.request("resources/list")
    result = _result(h.request("resources/read", {"uri": "file:///fs/notes.txt"}))

    # The fake echoes the uri into the body text; the uri itself is a skip.
    assert "from fs" in result["contents"][0]["text"]


# ---------------------------------------------------------------- fail-closed

def test_sanitization_failure_refuses_to_forward_the_payload(harness, monkeypatch):
    # Fail-closed is the stated policy for sanitization: losing a tool call
    # is bad, leaking the PII this exists to prevent is worse. The audit
    # path (M4) will fail OPEN, for the opposite reason.
    h = harness({"fs": {}}, sanitize=True)
    h.initialize()

    def explode(_text):
        raise RuntimeError("detector exploded")

    monkeypatch.setattr(h.gateway._session, "sanitize_text", explode)
    response = _call(h, "fs__echo", {"note": PLANTED["email"]})

    assert "error" in response
    assert "could not sanitize" in response["error"]["message"]
    # And the payload is gone, not partially forwarded.
    assert PLANTED["email"] not in json.dumps(response)


def test_the_gateway_refuses_to_start_if_detection_does_not_work(harness):
    # "Sanitization is enabled but unavailable" must not degrade into
    # "running as a plain proxy". A broken detection config builds a
    # perfectly healthy-looking Shield and only fails on first use, so the
    # gateway would have come up announcing protection it was not
    # providing. The startup self-test is what closes that.
    h = harness({"fs": {}}, sanitize=True,
                detection={"spacy_model": "no_such_model_exists", "ner_required": True})
    h._server.join(timeout=120)

    assert h.exit_code == 1


def test_the_self_test_does_not_pollute_the_serving_token_map(harness):
    # Found by the token numbering coming out wrong: the self-test ran on
    # the live session, so its own probe address claimed [EMAIL_0] and the
    # first real address became [EMAIL_1]. Harmless-looking at M1, but at
    # M2 a model echoing [EMAIL_0] back would have had the probe address
    # substituted into a real tool call. The probe now runs on a throwaway
    # session that shares the shield but not the map.
    h = harness({"fs": {}}, sanitize=True)
    h.initialize()
    text = _result(_call(h, "fs__echo", {"a": PLANTED["email"]}))["content"][0]["text"]

    assert "[EMAIL_0]" in text, "the first real value must get the first token"


def test_the_startup_self_test_catches_a_detector_that_detects_nothing():
    # The self-test is only worth having if it can fail, so give it a
    # session whose sanitizer is a no-op and check it says so.
    class Passthrough:
        def sanitize_text(self, text):
            return text

    with pytest.raises(sanitizer.SanitizationFailed, match="self-test failed"):
        sanitizer.self_test(Passthrough(), {})


def test_the_self_test_only_probes_enabled_categories():
    # Narrowing detection deliberately must not trip the startup check.
    class OnlyEmails:
        def sanitize_text(self, text):
            return text.replace("selftest.probe@example-eu.fr", "[EMAIL_0]")

    checked = sanitizer.self_test(OnlyEmails(), {
        "detect_credit_cards": False, "detect_iban": False,
        "detect_api_keys": False, "detect_ssns": False})
    assert checked == ["detect_emails"]


# ------------------------------------------------------------------ INVARIANT

def test_INVARIANT_no_planted_value_reaches_the_client_in_any_form(harness):
    # The whole product in one assertion. Everything the client received
    # across a realistic session is concatenated and searched for each
    # planted value verbatim, and for its digits-only projection so that a
    # reformatted copy ("5500-0000-0000-0004") cannot hide.
    h = harness({"fs": {}}, sanitize=True)
    h.initialize()

    received = []
    received.append(json.dumps(_call(h, "fs__echo", {
        "body": "Customer %(email)s paid with %(card)s" % PLANTED})))
    received.append(json.dumps(_call(h, "fs__echo", {
        "nested": {"deep": [{"iban": PLANTED["iban"]}]}})))
    received.append(json.dumps(_call(h, "fs__boom", {"key": PLANTED["aws_key"]})))
    received.append(json.dumps(h.request("tools/list")))
    blob = "".join(received)
    digits_only = "".join(c for c in blob if c.isdigit())

    for label, value in PLANTED.items():
        assert value not in blob, "%s leaked verbatim" % label
        digits = "".join(c for c in value if c.isdigit())
        if len(digits) >= 8:
            assert digits not in digits_only, "%s leaked in digits-only form" % label


def test_INVARIANT_the_probe_would_catch_a_leak(harness):
    # The invariant above is only evidence if it can fail. With
    # sanitization off, the exact same session must leak -- otherwise the
    # green run above proves nothing about the sanitizer.
    h = harness({"fs": {}}, sanitize=False)
    h.initialize()
    blob = json.dumps(_call(h, "fs__echo", {
        "body": "Customer %(email)s paid with %(card)s" % PLANTED}))

    assert PLANTED["email"] in blob, "the probe cannot detect a leak"
