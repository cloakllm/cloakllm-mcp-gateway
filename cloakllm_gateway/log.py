"""Diagnostic logging, on stderr only.

stdout is the protocol channel. One stray print() there corrupts a JSON-RPC
frame and the client sees a parse error it cannot attribute to anything. Every
diagnostic in this package goes through here, and here only writes to stderr.

Output is ASCII-only for the same reason the SDK's is: a Windows console
running a non-UTF-8 codepage (cp1255 on this project's dev machine) raises
UnicodeEncodeError on a non-ASCII write, which would take the gateway down
from inside a log line.
"""

import sys
import threading
import time

LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40, "silent": 100}

_lock = threading.Lock()
_threshold = LEVELS["info"]


def set_level(name):
    global _threshold
    _threshold = LEVELS.get(str(name).lower(), LEVELS["info"])


def _emit(level, prefix, message):
    if LEVELS[level] < _threshold:
        return
    line = "[cloakllm-gateway] %s %-5s %s\n" % (
        time.strftime("%H:%M:%S"), prefix, message)
    with _lock:
        try:
            sys.stderr.write(line.encode("ascii", "replace").decode("ascii"))
            sys.stderr.flush()
        except (OSError, ValueError):
            pass


def debug(message):
    _emit("debug", "DEBUG", message)


def info(message):
    _emit("info", "INFO", message)


def warn(message):
    _emit("warn", "WARN", message)


def error(message):
    _emit("error", "ERROR", message)
