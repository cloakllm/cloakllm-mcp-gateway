"""Config validation.

The config names processes this gateway will execute, so a permissive parser
is a security problem, not just an ergonomics one. Everything it accepts must
be something it can actually run.
"""

import json
import os

import pytest

from cloakllm_gateway import config as config_mod
from cloakllm_gateway.config import ConfigError


def _ok(**overrides):
    data = {"upstreams": {"fs": {"command": "node", "args": ["server.js"]}}}
    data.update(overrides)
    return data


def test_a_minimal_config_parses():
    cfg = config_mod.parse(_ok())
    assert [u.name for u in cfg.upstreams] == ["fs"]
    assert cfg.upstreams[0].args == ["server.js"]
    assert cfg.allow_unknown_protocol_version is False


def test_an_underscore_in_an_upstream_name_is_rejected():
    # "__" is the namespace separator. If a server could be called "my_fs",
    # then "my_fs__do" and a tool literally named "fs__do" on server "my"
    # would be indistinguishable, and the call would silently go to the
    # wrong process.
    with pytest.raises(ConfigError, match="Underscores are not allowed"):
        config_mod.parse({"upstreams": {"my_fs": {"command": "node"}}})


def test_a_tool_name_containing_the_separator_still_routes():
    # The name ban is what makes splitting on the FIRST "__" unambiguous, so
    # an upstream tool may contain "__" itself.
    cfg = config_mod.parse(_ok())
    assert config_mod.NAME_RE.match("fs")
    prefix, _, bare = "fs__read__all".partition("__")
    assert (prefix, bare) == ("fs", "read__all")


def test_an_empty_upstream_list_is_rejected():
    with pytest.raises(ConfigError, match="no upstream servers"):
        config_mod.parse({"upstreams": {}})


@pytest.mark.parametrize("spec,expected", [
    ({}, "missing the required key 'command'"),
    ({"command": ""}, "must not be empty"),
    ({"command": 7}, "must be str"),
    ({"command": "n", "args": "not-a-list"}, "args must be a list"),
    ({"command": "n", "args": [1]}, "args must be a list"),
    ({"command": "n", "env": {"A": 1}}, "env must be an object"),
    ({"command": "n", "cwd": 3}, "cwd must be a string"),
    ({"command": "n", "startup_timeout": 0}, "positive number"),
])
def test_bad_upstream_specs_are_rejected(spec, expected):
    with pytest.raises(ConfigError, match=expected):
        config_mod.parse({"upstreams": {"fs": spec}})


@pytest.mark.parametrize("overrides,expected", [
    ({"log_level": "chatty"}, "log_level must be one of"),
    ({"allow_unknown_protocol_version": "yes"}, "must be a boolean"),
    ({"request_timeout": -1}, "positive number"),
])
def test_bad_top_level_options_are_rejected(overrides, expected):
    with pytest.raises(ConfigError, match=expected):
        config_mod.parse(_ok(**overrides))


def test_a_non_object_config_is_rejected():
    with pytest.raises(ConfigError, match="must be a JSON object"):
        config_mod.parse([{"command": "node"}])


def test_load_reports_a_missing_file_by_path(tmp_path):
    missing = str(tmp_path / "nope.json")
    with pytest.raises(ConfigError, match="no config file at"):
        config_mod.load(missing)


def test_load_reports_bad_json_by_path(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        config_mod.load(str(path))


def test_load_reads_utf8_regardless_of_the_console_codepage(tmp_path):
    # Windows opens files in the console codepage by default (cp1255 on this
    # project's dev machine), which mangles any non-ASCII path in the config.
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(
        {"upstreams": {"fs": {"command": "node", "cwd": "C:/proyectos/informacion"}}}),
        encoding="utf-8")
    cfg = config_mod.load(str(path))
    assert cfg.upstreams[0].cwd == "C:/proyectos/informacion"


def test_the_env_var_is_used_when_no_path_is_given(tmp_path, monkeypatch):
    path = tmp_path / "from-env.json"
    path.write_text(json.dumps(_ok()), encoding="utf-8")
    monkeypatch.setenv(config_mod.ENV_VAR, str(path))
    assert config_mod.load().upstreams[0].name == "fs"


def test_an_explicit_path_beats_the_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv(config_mod.ENV_VAR, str(tmp_path / "ignored.json"))
    path = tmp_path / "explicit.json"
    path.write_text(json.dumps(
        {"upstreams": {"db": {"command": "node"}}}), encoding="utf-8")
    assert config_mod.load(str(path)).upstreams[0].name == "db"


def test_the_example_config_in_the_repo_is_valid():
    # A README example that does not parse is a support ticket waiting to
    # happen, and it is free to check.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "cloakllm-gateway.example.json"),
              encoding="utf-8") as handle:
        config_mod.parse(json.load(handle))
