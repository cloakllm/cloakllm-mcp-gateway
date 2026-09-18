"""M0: the gateway carries real traffic without the client noticing.

M0 ships no sanitization on purpose. Its whole value is that when M1 starts
rewriting payloads there is a proven-transparent baseline to diff against --
so any difference the client can observe from here on is a change M1 made
deliberately, not plumbing damage.
"""

import copy

import pytest

# Imported as "conftest", not "tests.conftest": pytest loads conftest.py as a
# top-level module, so importing it by package path creates a *second* module
# object whose Timeout is a different class from the one actually raised.
from conftest import PROTOCOL, Timeout


def _result(response):
    assert "error" not in response, response["error"]
    return response["result"]


# ----------------------------------------------------------------- handshake

def test_initialize_presents_the_gateway_not_an_upstream(harness):
    h = harness({"fs": {}})
    result = _result(h.initialize())

    assert result["serverInfo"]["name"] == "cloakllm-gateway"
    assert result["protocolVersion"] == PROTOCOL
    # A tool whose product is an audit trail should not claim to be something
    # else during its own handshake, however transparent it is otherwise.
    assert "fake-fs" not in str(result["serverInfo"])


def test_capabilities_are_the_union_of_the_upstreams(harness):
    h = harness({"fs": {}, "db": {}})
    caps = _result(h.initialize())["capabilities"]

    assert caps["tools"]["listChanged"] is True
    assert caps["resources"]["subscribe"] is True
    assert "prompts" in caps and "logging" in caps


def test_instructions_from_every_upstream_reach_the_client(harness):
    h = harness({"fs": {}, "db": {}})
    instructions = _result(h.initialize())["instructions"]

    assert "Instructions from fs." in instructions
    assert "Instructions from db." in instructions
    # The client is told how the namespacing works, or the model has no way
    # to know why every tool name suddenly has a prefix.
    assert "__" in instructions


def test_an_unknown_protocol_version_is_refused_loudly(harness):
    h = harness({"fs": {}})
    response = h.initialize(protocol="2099-01-01")

    assert "error" in response
    assert "2099-01-01" in response["error"]["message"]
    # Loud means actionable: the operator is told what would work.
    assert PROTOCOL in response["error"]["data"]["supported"]


def test_an_unknown_protocol_version_can_be_forced(harness):
    # The escape hatch exists so an operator is not bricked by a spec bump,
    # but it is off by default and it warns.
    h = harness({"fs": {"protocol": "2099-01-01"}},
                allow_unknown_protocol_version=True)
    result = _result(h.initialize(protocol="2099-01-01"))

    assert result["protocolVersion"] == "2099-01-01"


def test_upstreams_on_different_protocol_versions_are_refused(harness):
    # Presenting one version to the client while speaking another to an
    # upstream would make the gateway a silent translator between dialects,
    # which is exactly the degradation this design refuses.
    h = harness({"fs": {}, "db": {"protocol": "2024-11-05"}})
    response = h.initialize()

    assert "error" in response
    assert "different protocol versions" in response["error"]["message"]


def test_requests_before_initialize_are_refused(harness):
    h = harness({"fs": {}})
    response = h.request("tools/list")

    assert "error" in response
    assert "before initialize" in response["error"]["message"]


def test_ping_is_answered_locally_even_before_initialize(harness):
    h = harness({"fs": {}})
    assert _result(h.request("ping")) == {}


# --------------------------------------------------------------------- tools

def test_tools_list_merges_upstreams_and_follows_pagination(harness):
    h = harness({"fs": {}, "db": {}})
    h.initialize()
    tools = _result(h.request("tools/list"))["tools"]
    names = sorted(t["name"] for t in tools)

    # Four tools per upstream, and they only all appear if the gateway
    # followed nextCursor instead of stopping at the first page.
    assert names == [
        "db__ask_client", "db__boom", "db__echo", "db__slow",
        "fs__ask_client", "fs__boom", "fs__echo", "fs__slow",
    ]


def test_unknown_fields_on_a_tool_survive_the_proxy(harness):
    # The LlamaIndex E1 lesson at the schema layer: a proxy that parses into
    # a typed model drops every field the model has not heard of, and the
    # client cannot see what went missing.
    h = harness({"fs": {}})
    h.initialize()
    tools = _result(h.request("tools/list"))["tools"]
    echo = next(t for t in tools if t["name"] == "fs__echo")

    assert echo["x-vendor-extension"] == {"deep": ["preserve", "me"]}


def test_a_tool_call_is_routed_and_de_namespaced(harness):
    h = harness({"fs": {}, "db": {}})
    h.initialize()
    result = _result(h.request("tools/call", {
        "name": "db__echo", "arguments": {"a": 1}}))

    # The upstream saw the bare name, and the right upstream answered.
    assert "db echoes" in result["content"][0]["text"]
    assert result["x-result-extension"] == "preserve me too"


def test_a_tool_error_reaches_the_client_intact(harness):
    h = harness({"fs": {}})
    h.initialize()
    response = h.request("tools/call", {"name": "fs__boom", "arguments": {}})

    assert response["error"]["code"] == -32000
    assert response["error"]["message"] == "boom from fs"


@pytest.mark.parametrize("name,expected", [
    ("nosuchserver__echo", "no upstream named"),
    ("echo", "not namespaced"),
])
def test_unroutable_tool_names_fail_clearly(harness, name, expected):
    h = harness({"fs": {}})
    h.initialize()
    response = h.request("tools/call", {"name": name, "arguments": {}})

    assert expected in response["error"]["message"]


def test_an_unknown_method_is_refused_rather_than_guessed(harness):
    h = harness({"fs": {}})
    h.initialize()
    response = h.request("experimental/whatever", {})

    assert response["error"]["code"] == -32601
    assert "not proxied" in response["error"]["message"]


# ----------------------------------------------------------------- resources

def test_resource_uris_are_not_namespaced(harness):
    # A URI means something to the server that issued it and is shown to the
    # user. Rewriting it would be a visible lie, so routing uses an index
    # built from resources/list instead.
    h = harness({"fs": {}, "db": {}})
    h.initialize()
    uris = sorted(r["uri"] for r in _result(h.request("resources/list"))["resources"])

    assert uris == ["file:///db/notes.txt", "file:///fs/notes.txt"]


def test_a_resource_read_routes_by_the_indexed_uri(harness):
    h = harness({"fs": {}, "db": {}})
    h.initialize()
    h.request("resources/list")
    result = _result(h.request("resources/read", {"uri": "file:///db/notes.txt"}))

    assert "from db" in result["contents"][0]["text"]


def test_an_unindexed_uri_is_ambiguous_across_two_providers(harness):
    h = harness({"fs": {}, "db": {}})
    h.initialize()
    response = h.request("resources/read", {"uri": "file:///never/listed"})

    assert "cannot tell which upstream owns" in response["error"]["message"]


def test_an_unindexed_uri_resolves_when_only_one_upstream_can_own_it(harness):
    h = harness({"fs": {}})
    h.initialize()
    result = _result(h.request("resources/read", {"uri": "file:///anything"}))

    assert "from fs" in result["contents"][0]["text"]


def test_a_primitive_no_upstream_offers_is_method_not_found(harness):
    # Found by running the real @modelcontextprotocol/server-filesystem
    # beside the gateway and diffing the transcripts: the real server
    # answers -32601 for resources/list, and the gateway was synthesising an
    # empty list. "There are none" and "I do not implement that" are
    # different statements, and the fake upstream -- which implements
    # everything -- could never have surfaced it.
    h = harness({"fs": {"tools_only": True}})
    h.initialize()
    response = h.request("resources/list")

    assert response["error"]["code"] == -32601
    assert "no upstream provides resources" in response["error"]["message"]


def test_resource_templates_are_aggregated(harness):
    h = harness({"fs": {}, "db": {}})
    h.initialize()
    templates = _result(h.request("resources/templates/list"))["resourceTemplates"]

    assert len(templates) == 2


# ------------------------------------------------------------------- prompts

def test_prompts_are_namespaced_and_routed(harness):
    h = harness({"fs": {}, "db": {}})
    h.initialize()
    names = sorted(p["name"] for p in _result(h.request("prompts/list"))["prompts"])
    assert names == ["db__greet", "fs__greet"]

    result = _result(h.request("prompts/get", {
        "name": "fs__greet", "arguments": {"who": "world"}}))
    assert "hello world from fs" in result["messages"][0]["content"]["text"]


def test_completion_is_routed_by_its_prompt_ref(harness):
    h = harness({"fs": {}, "db": {}})
    h.initialize()
    result = _result(h.request("completion/complete", {
        "ref": {"type": "ref/prompt", "name": "db__greet"},
        "argument": {"name": "who", "value": "w"}}))

    value = result["completion"]["values"][0]
    assert "completed by db" in value
    # The upstream must have seen its own bare prompt name, not ours.
    assert "db__greet" not in value


def test_logging_set_level_reaches_every_upstream(harness):
    h = harness({"fs": {}, "db": {}})
    h.initialize()
    assert _result(h.request("logging/setLevel", {"level": "debug"})) == {}


# ------------------------------------------------------ bidirectional traffic

def test_an_upstream_can_ask_the_client_a_question(harness):
    # sampling/createMessage travels server -> client, so ids have to be
    # remapped in the other direction too. Two upstreams both starting at
    # id 1 is the collision this prevents.
    h = harness({"fs": {}})
    h.initialize()
    h.send({"jsonrpc": "2.0", "id": 99, "method": "tools/call",
            "params": {"name": "fs__ask_client", "arguments": {}}})

    ask = h.await_request("sampling/createMessage")
    assert ask["params"]["from"] == "fs"
    # The client must not see the upstream's own id.
    assert ask["id"] != "srv-1"

    h.send({"jsonrpc": "2.0", "id": ask["id"],
            "result": {"role": "assistant", "content": {"type": "text", "text": "ok"}}})

    result = _result(h.await_response(99))
    assert "ok" in result["content"][0]["text"]


def test_two_upstreams_asking_at_once_do_not_collide(harness):
    h = harness({"fs": {}, "db": {}})
    h.initialize()
    h.send({"jsonrpc": "2.0", "id": "a", "method": "tools/call",
            "params": {"name": "fs__ask_client", "arguments": {}}})
    h.send({"jsonrpc": "2.0", "id": "b", "method": "tools/call",
            "params": {"name": "db__ask_client", "arguments": {}}})

    first = h.await_request("sampling/createMessage")
    second = h.await_request("sampling/createMessage")
    assert first["id"] != second["id"], "both upstreams used id srv-1 internally"

    for ask in (first, second):
        # Answer each with a value naming the upstream that asked. If the
        # reverse routing crossed the two, an upstream gets the other's
        # answer and the assertions below catch it.
        h.send({"jsonrpc": "2.0", "id": ask["id"],
                "result": {"answered": ask["params"]["from"]}})

    assert {first["params"]["from"], second["params"]["from"]} == {"fs", "db"}
    assert "fs" in _result(h.await_response("a"))["content"][0]["text"]
    assert "db" in _result(h.await_response("b"))["content"][0]["text"]


def test_a_cancellation_is_rewritten_to_the_id_the_upstream_knows(harness):
    # The upstream only answers "slow" once it sees a cancellation carrying
    # the id *it* was given -- which is the gateway's id, not the client's.
    # A forwarded-verbatim requestId would leave this hanging.
    h = harness({"fs": {}})
    h.initialize()
    h.send({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {"name": "fs__slow", "arguments": {}}})
    h.notify("notifications/cancelled", {"requestId": 7, "reason": "user"})

    result = _result(h.await_response(7))
    assert result["content"][0]["text"] == "cancelled"


def test_a_string_id_and_an_integer_id_do_not_alias(harness):
    # JSON-RPC allows both, and 1 must not be confused with "1".
    h = harness({"fs": {}})
    h.initialize()
    h.send({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "fs__echo", "arguments": {"which": "int"}}})
    h.send({"jsonrpc": "2.0", "id": "1", "method": "tools/call",
            "params": {"name": "fs__echo", "arguments": {"which": "str"}}})

    as_int = _result(h.await_response(1))
    as_str = _result(h.await_response("1"))
    assert "int" in as_int["content"][0]["text"]
    assert "str" in as_str["content"][0]["text"]


# ------------------------------------------------------------------- hygiene

def test_upstream_stderr_never_reaches_the_protocol_stream(harness):
    # One diagnostic line on stdout corrupts a frame and the client sees a
    # parse error it cannot attribute to anything.
    h = harness({"fs": {"noisy_stderr": True}})
    h.initialize()
    h.request("tools/list")

    assert not [m for m in h.drain() if "__raw__" in m], \
        "something non-JSON reached the client's stdin"


def test_an_upstream_that_dies_fails_its_in_flight_calls(harness):
    # Without this the client waits forever on a process that is already
    # gone, which in a desktop client looks like the model hanging.
    h = harness({"fs": {}})
    h.initialize()
    h.send({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": {"name": "fs__slow", "arguments": {}}})
    # Let the call reach the upstream before pulling the process out.
    h.gateway.upstreams["fs"].request("ping", {}, timeout=10)
    h.gateway.upstreams["fs"]._proc.kill()

    response = h.await_response(5)
    assert "error" in response
    assert "exited" in response["error"]["message"]


def test_an_upstream_that_cannot_start_stops_the_gateway(harness):
    # Half a gateway is worse than none: the client would get a tool list
    # missing whatever the dead upstream provided, with nothing to show why.
    h = harness({"fs": {}, "dead": {"command": "definitely-not-a-real-binary"}})
    h._server.join(timeout=20)

    assert h.exit_code == 1


def test_the_client_message_is_not_mutated_in_place(harness):
    # Every rewrite copies. If the gateway mutated the incoming dict, a
    # second reader of the same object -- an audit hook at M4, say -- would
    # see the rewritten form and log the wrong thing.
    h = harness({"fs": {}})
    h.initialize()
    msg = {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
           "params": {"name": "fs__echo", "arguments": {"x": 1}}}
    before = copy.deepcopy(msg)
    h.send(msg)
    _result(h.await_response(3))

    assert msg == before


def test_a_malformed_client_line_gets_a_parse_error_and_the_loop_survives(harness):
    h = harness({"fs": {}})
    h.initialize()
    h._to_gateway.write(b"{not json at all\n")
    h._to_gateway.flush()

    error = h._next(lambda m: "error" in m and m.get("id") is None, 10)
    assert error["error"]["code"] == -32700
    # And the gateway is still serving.
    assert _result(h.request("ping")) == {}


def test_more_concurrent_list_calls_than_worker_threads(harness):
    # Found by reading the threading, not by a failure: _handle_list runs on
    # the handler pool and its fan-out used to submit to that same pool, so
    # enough simultaneous list calls would fill the pool with handlers all
    # waiting on work that could never be scheduled. The gateway hung for
    # good -- no error, no timeout, just silence.
    h = harness({"fs": {}})
    h.initialize()
    pool_size = h.gateway._workers._max_workers
    ids = list(range(100, 100 + pool_size * 3))
    for msg_id in ids:
        h.send({"jsonrpc": "2.0", "id": msg_id, "method": "tools/list"})

    for msg_id in ids:
        assert "result" in h.await_response(msg_id, timeout=30)


def test_timeout_helper_actually_times_out(harness):
    # The negative assertions above are only worth anything if the harness
    # can fail. A helper that never times out would make them all vacuous.
    h = harness({"fs": {}})
    h.initialize()
    with pytest.raises(Timeout):
        h.await_notification("notifications/nothing-sends-this", timeout=0.5)
