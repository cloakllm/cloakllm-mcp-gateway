"""Command line entry point.

Run as an MCP server over stdio, spawning the upstreams named in the config:

    cloakllm-gateway --config ./cloakllm-gateway.json
"""

import argparse
import sys

from . import __version__, config as config_mod, log
from .gateway import Gateway


def _binary_stdio():
    """The raw byte streams, bypassing text-mode newline translation.

    On Windows, text mode turns every "\\n" written into "\\r\\n", which is
    not what newline-delimited JSON-RPC framing expects, and it decodes
    incoming bytes with the console codepage instead of UTF-8.
    """
    stdin = getattr(sys.stdin, "buffer", sys.stdin)
    stdout = getattr(sys.stdout, "buffer", sys.stdout)
    return stdin, stdout


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="cloakllm-gateway",
        description="A CloakLLM proxy between an MCP client and other MCP servers.")
    parser.add_argument("--config", help="path to the gateway config JSON")
    parser.add_argument("--log-level", choices=sorted(log.LEVELS),
                        help="override the config's log_level")
    parser.add_argument("--check", action="store_true",
                        help="validate the config and exit without serving")
    parser.add_argument("--version", action="version",
                        version="cloakllm-gateway %s" % __version__)
    args = parser.parse_args(argv)

    if args.log_level:
        log.set_level(args.log_level)

    try:
        cfg = config_mod.load(args.config)
    except config_mod.ConfigError as exc:
        log.error(str(exc))
        return 2

    if not args.log_level:
        log.set_level(cfg.log_level)

    if args.check:
        log.info("config is valid: %d upstream(s): %s"
                 % (len(cfg.upstreams),
                    ", ".join(sorted(u.name for u in cfg.upstreams))))
        return 0

    stdin, stdout = _binary_stdio()
    gateway = Gateway(cfg, stdin, stdout)
    try:
        return gateway.run()
    except KeyboardInterrupt:
        gateway.shutdown()
        return 0


if __name__ == "__main__":
    sys.exit(main())
