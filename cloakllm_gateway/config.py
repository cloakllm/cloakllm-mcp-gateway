"""Gateway configuration.

The config names the upstream MCP servers the gateway spawns, so it is a list
of commands this process will execute. Treat it exactly as you would treat a
Claude Desktop or Cursor MCP config: it must come from a trusted location on
disk, never from the network and never from anything an upstream server said.
Nothing in a server's responses can influence what gets spawned.
"""

import json
import os
import re

from . import log

DEFAULT_FILENAME = "cloakllm-gateway.json"
ENV_VAR = "CLOAKLLM_GATEWAY_CONFIG"

# Upstream names become the prefix in a namespaced tool name, joined by "__".
# Forbidding "_" in the name itself is what makes splitting on the first "__"
# unambiguous when the tool's own name also contains one -- "fs__read__all"
# can only be tool "read__all" on upstream "fs".
NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9-]*$")

# Several clients reject tool names longer than this.
MAX_TOOL_NAME = 128


class ConfigError(Exception):
    """The config file is missing, unreadable, or invalid."""


class UpstreamSpec:
    __slots__ = ("name", "command", "args", "env", "cwd", "startup_timeout")

    def __init__(self, name, command, args=None, env=None, cwd=None,
                 startup_timeout=30.0):
        self.name = name
        self.command = command
        self.args = list(args or [])
        self.env = dict(env or {})
        self.cwd = cwd
        self.startup_timeout = startup_timeout

    def __repr__(self):
        return "UpstreamSpec(%r, %r)" % (self.name, self.command)


# Detection options an operator may set. Deliberately a subset of
# ShieldConfig: the fields it leaves out (audit_enabled, log_dir,
# attestation_key, compliance_mode, ...) are the gateway's to own, and an
# operator silently redirecting the SDK's audit chain from a proxy config
# would be a surprising way to lose a log.
DETECTION_OPTIONS = frozenset({
    "locale", "spacy_model", "ner_entity_types", "ner_required",
    "detect_emails", "detect_phones", "detect_ssns", "detect_credit_cards",
    "detect_api_keys", "detect_ip_addresses", "detect_iban",
    "custom_patterns", "mode", "descriptive_tokens", "max_input_length",
})


class GatewayConfig:
    def __init__(self, upstreams, log_level="info",
                 allow_unknown_protocol_version=False,
                 request_timeout=120.0, sanitize=True, detection=None,
                 token_scope="session"):
        self.upstreams = upstreams
        self.log_level = log_level
        self.allow_unknown_protocol_version = allow_unknown_protocol_version
        self.request_timeout = request_timeout
        self.sanitize = sanitize
        self.detection = detection or {}
        self.token_scope = token_scope


def _require(obj, key, kind, where):
    if key not in obj:
        raise ConfigError("%s is missing the required key %r" % (where, key))
    value = obj[key]
    if not isinstance(value, kind):
        raise ConfigError("%s: %r must be %s, got %s"
                          % (where, key, kind.__name__, type(value).__name__))
    return value


def parse(data):
    """Validate a decoded config object into a GatewayConfig."""
    if not isinstance(data, dict):
        raise ConfigError("config must be a JSON object")

    raw = _require(data, "upstreams", dict, "config")
    if not raw:
        raise ConfigError("config lists no upstream servers; "
                          "a gateway with nothing behind it does nothing")

    upstreams = []
    for name, spec in raw.items():
        where = "upstream %r" % name
        if not NAME_RE.match(name):
            raise ConfigError(
                "%s: name must match %s. Underscores are not allowed because "
                "the namespace separator is '__'." % (where, NAME_RE.pattern))
        if not isinstance(spec, dict):
            raise ConfigError("%s: must be an object" % where)

        command = _require(spec, "command", str, where)
        if not command.strip():
            raise ConfigError("%s: command must not be empty" % where)

        args = spec.get("args", [])
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise ConfigError("%s: args must be a list of strings" % where)

        env = spec.get("env", {})
        if not isinstance(env, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise ConfigError("%s: env must be an object of string to string" % where)

        cwd = spec.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise ConfigError("%s: cwd must be a string" % where)

        timeout = spec.get("startup_timeout", 30.0)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ConfigError("%s: startup_timeout must be a positive number" % where)

        upstreams.append(UpstreamSpec(name, command, args, env, cwd, float(timeout)))

    log_level = data.get("log_level", "info")
    if log_level not in log.LEVELS:
        raise ConfigError("log_level must be one of %s"
                          % ", ".join(sorted(log.LEVELS)))

    allow_unknown = data.get("allow_unknown_protocol_version", False)
    if not isinstance(allow_unknown, bool):
        raise ConfigError("allow_unknown_protocol_version must be a boolean")

    request_timeout = data.get("request_timeout", 120.0)
    if not isinstance(request_timeout, (int, float)) or request_timeout <= 0:
        raise ConfigError("request_timeout must be a positive number")

    sanitize = data.get("sanitize", True)
    if not isinstance(sanitize, bool):
        raise ConfigError("sanitize must be a boolean")

    detection = data.get("detection", {})
    if not isinstance(detection, dict):
        raise ConfigError("detection must be an object")
    # Reject unknown keys rather than ignoring them. A typo in a detection
    # setting is the worst kind of silent failure here: the operator
    # believes a category is switched on and it is not.
    unknown = sorted(set(detection) - DETECTION_OPTIONS)
    if unknown:
        raise ConfigError(
            "unknown detection option(s): %s. Supported: %s"
            % (", ".join(unknown), ", ".join(sorted(DETECTION_OPTIONS))))
    if "ner_entity_types" in detection:
        detection = dict(detection, ner_entity_types=set(detection["ner_entity_types"]))

    # "session": a token minted from one upstream's data can be restored in
    # a call to any other. That is what lets a model read with one tool and
    # write with another, and it is also how the gateway can be made to hand
    # one server's data to a different one. "upstream" refuses that.
    token_scope = data.get("token_scope", "session")
    if token_scope not in ("session", "upstream"):
        raise ConfigError('token_scope must be "session" or "upstream"')

    return GatewayConfig(upstreams, log_level, allow_unknown,
                         float(request_timeout), sanitize, detection,
                         token_scope)


def find_path(explicit=None):
    """Resolve the config path: --config, then the env var, then the cwd."""
    if explicit:
        return explicit
    from_env = os.environ.get(ENV_VAR)
    if from_env:
        return from_env
    return os.path.join(os.getcwd(), DEFAULT_FILENAME)


def load(path=None):
    path = find_path(path)
    if not os.path.isfile(path):
        raise ConfigError(
            "no config file at %s. Point at one with --config, or set %s."
            % (path, ENV_VAR))
    try:
        # Explicit utf-8: the Windows default is the console codepage, which
        # mangles any non-ASCII path or argument in the config.
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except ValueError as exc:
        raise ConfigError("%s is not valid JSON: %s" % (path, exc)) from exc
    except OSError as exc:
        raise ConfigError("cannot read %s: %s" % (path, exc)) from exc
    return parse(data)
