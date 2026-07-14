"""LLM completion backends for the LLM-based classification pipeline.

Two backends, resolved automatically by :func:`get_backend`:

* :class:`AnthropicBackend` -- the official ``anthropic`` SDK. Used whenever
  the SDK can resolve credentials (``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN``,
  or an ``ant auth login`` profile).
* :class:`ClaudeCLIBackend` -- shells out to the Claude Code binary in headless
  mode (``claude -p``), which authenticates via the user's Claude Code login.
  Used as a fallback so the pipeline runs without an API key.

Both expose ``complete(system, user) -> str`` returning the model's text.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

DEFAULT_MODEL = os.environ.get("SOCRATIC_LLM_MODEL", "claude-opus-4-8")


class AnthropicBackend:
    """Direct Messages API calls via the official SDK."""

    def __init__(self, model: str = DEFAULT_MODEL):
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic()

    def complete(self, system: str, user: str) -> str:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=8192,
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        ) as stream:
            message = stream.get_final_message()
        if message.stop_reason == "refusal":
            raise RuntimeError("model refused the request")
        return "".join(b.text for b in message.content if b.type == "text")


class ClaudeCLIBackend:
    """Headless ``claude -p`` calls (uses the user's Claude Code login)."""

    def __init__(self, model: str = DEFAULT_MODEL, binary: str | None = None):
        self.model = model
        self.binary = binary or find_claude_binary()
        if not self.binary:
            raise RuntimeError("claude binary not found")

    def complete(self, system: str, user: str) -> str:
        prompt = f"{system}\n\n{user}"
        proc = subprocess.run(
            [
                self.binary, "-p", prompt,
                "--model", self.model,
                "--output-format", "json",
                "--max-turns", "1",
            ],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            detail = (proc.stderr.strip() or proc.stdout.strip())[-500:]
            raise RuntimeError(f"claude CLI failed (exit {proc.returncode}): {detail}")
        data = json.loads(proc.stdout)
        if data.get("is_error"):
            raise RuntimeError(f"claude CLI error result: {data.get('result', '')[:500]}")
        return data["result"]


def find_claude_binary() -> str | None:
    """Locate a Claude Code binary (PATH, env, or the VS Code extension)."""
    for candidate in (os.environ.get("CLAUDE_CODE_EXECPATH"), shutil.which("claude")):
        if candidate and Path(candidate).exists():
            return candidate
    ext_dir = Path.home() / ".vscode" / "extensions"
    if ext_dir.is_dir():
        hits = sorted(ext_dir.glob("anthropic.claude-code-*/resources/native-binary/claude"))
        if hits:
            return str(hits[-1])
    return None


def get_backend(model: str = DEFAULT_MODEL):
    """Prefer the SDK when credentials resolve; otherwise the claude CLI."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return AnthropicBackend(model)
    try:
        return ClaudeCLIBackend(model)
    except RuntimeError:
        # Last resort: let the SDK try its own credential resolution
        # (e.g. an `ant auth login` profile on disk).
        return AnthropicBackend(model)
