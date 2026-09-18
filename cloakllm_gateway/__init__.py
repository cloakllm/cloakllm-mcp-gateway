"""CloakLLM MCP Gateway -- a sanitizing proxy in front of other MCP servers.

Deliberately off the SDK version line: this is a product surface, not part of
the cloakllm / cloakllm-js / cloakllm-mcp alignment, and it must be free to
release on its own cadence.
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
