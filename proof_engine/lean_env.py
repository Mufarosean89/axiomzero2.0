"""Axiom Zero — Lean 4 Environment Manager

Manages a Lean 4 server subprocess and communicates via the Language Server
Protocol (LSP) over JSON-RPC 2.0.

Protocol:
1. Spawn ``lean --server`` as a subprocess (binary pipes).
2. Send ``Content-Length: N\\r\\n\\r\\n{json}`` frames to stdin.
3. Read ``Content-Length: N\\r\\n\\r\\n{json}`` frames from stdout.
4. LSP handshake: initialize -> textDocument/didOpen -> ...
5. Lean-specific RPC: ``$/lean/rpc/call`` with ``getInteractiveGoals``.
6. File changes via ``textDocument/didChange``.
7. Shutdown on completion.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .proof_state import ProofState, Goal, Hypothesis, GoalStatus, TacticStep


# ── LSP Protocol Helpers ────────────────────────────────────────────────────

def _encode_lsp_message(msg: Dict[str, Any]) -> bytes:
    """Encode a JSON-RPC message as an LSP frame with Content-Length header."""
    body = json.dumps(msg, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def _parse_lsp_response(data: bytes) -> tuple[list[Dict[str, Any]], int]:
    """Parse LSP frames from a byte buffer. Returns (messages, bytes_consumed)."""
    messages: list[Dict[str, Any]] = []
    bytes_consumed = 0
    remaining = data

    while remaining:
        header_match = re.match(
            rb"Content-Length:\s*(\d+)\s*\r?\n\r?\n",
            remaining,
            re.IGNORECASE,
        )
        if not header_match:
            break

        content_length = int(header_match.group(1))
        header_end = header_match.end()
        body_end = header_end + content_length

        if body_end > len(remaining):
            break

        body = remaining[header_end:body_end]
        try:
            messages.append(json.loads(body))
        except json.JSONDecodeError:
            pass

        bytes_consumed += body_end
        remaining = remaining[body_end:]

    return messages, bytes_consumed


class LSPError(Exception):
    """Error from an LSP JSON-RPC response."""

    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"LSP error {code}: {message}")


class LeanServerError(Exception):
    """Error from the Lean 4 server."""
    pass


# ── JSON-RPC 2.0 Helpers ────────────────────────────────────────────────────

_request_counter = 0


def _make_request(method: str, params: Any = None) -> Dict[str, Any]:
    """Create a JSON-RPC 2.0 request dict with auto-incremented id."""
    global _request_counter
    _request_counter += 1
    msg: Dict[str, Any] = {"jsonrpc": "2.0", "id": _request_counter, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def _make_notification(method: str, params: Any = None) -> Dict[str, Any]:
    """Create a JSON-RPC 2.0 notification dict (no id field)."""
    msg: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    return msg


# ── Lean 4 Environment ──────────────────────────────────────────────────────

class LeanEnv:
    """Manages a Lean 4 server subprocess using LSP over stdio.

    Usage::

        with LeanEnv() as env:
            env.open_file("theorem hello : 1 + 1 = 2 := by\\n  sorry\\n")
            env.apply_tactic("rfl")
            result = env.get_goals()
    """

    def __init__(
        self,
        lean_path: str = "lean",
        workspace_dir: Optional[str] = None,
        timeout: float = 30.0,
        verbose: bool = False,
    ):
        self.lean_path = lean_path
        self.timeout = timeout
        self.verbose = verbose

        self._process: Optional[subprocess.Popen] = None
        self._started = False
        self._request_id = 0
        self._rcv_buffer = b""
        self._rcv_lock = threading.Lock()
        self._rcv_event = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None

        self._workspace_dir = workspace_dir or tempfile.mkdtemp(prefix="axiom_zero_lean_")
        self._current_file: Optional[str] = None
        self._current_file_content = ""
        self._lean_version: Optional[str] = None
        self._server_capabilities: Dict[str, Any] = {}
        self._pending_messages: List[Dict[str, Any]] = []
        self._doc_version = 0

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Start the Lean 4 server and perform LSP initialize handshake."""
        if self._started:
            return True

        self._ensure_lean_available()

        Path(self._workspace_dir).mkdir(parents=True, exist_ok=True)

        try:
            self._process = subprocess.Popen(
                [self.lean_path, "--server"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self._workspace_dir,
            )
        except FileNotFoundError:
            raise LeanServerError(f"Could not execute Lean 4: {self.lean_path}")

        time.sleep(0.5)

        if self._process.poll() is not None:
            stderr = self._read_stderr() or ""
            raise LeanServerError(f"Lean server exited immediately. Error: {stderr[:500]}")

        self._started = True
        self._start_reader_thread()
        self._send_initialize()
        self._log("Lean 4 server started successfully")
        return True

    def stop(self) -> None:
        """Shut down the Lean 4 server gracefully."""
        if not self._started or not self._process:
            return
        try:
            self._send_notification("shutdown")
            self._send_notification("exit")
        except Exception:
            pass
        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except Exception:
            if self._process:
                self._process.kill()

        self._started = False
        self._process = None
        self._rcv_event.set()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2)
        self._reader_thread = None
        self._log("Lean 4 server stopped")

    def restart(self) -> bool:
        """Restart the Lean 4 server."""
        self.stop()
        time.sleep(0.3)
        return self.start()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ── File Management ──────────────────────────────────────────────────

    def open_file(self, content: str, module_name: str = "Temp") -> str:
        """Create a .lean file and open it on the server via textDocument/didOpen."""
        file_path = os.path.join(self._workspace_dir, f"{module_name}.lean")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        self._current_file = file_path
        self._current_file_content = content
        uri = self._path_to_uri(file_path)
        self._send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": "lean4",
                "version": 1,
                "text": content,
            }
        })
        self._log(f"Opened file: {file_path}")
        return file_path

    def update_file(self, new_content: str) -> None:
        """Update the current file on the server via textDocument/didChange."""
        if not self._current_file:
            raise LeanServerError("No file open. Call open_file() first.")
        uri = self._path_to_uri(self._current_file)
        self._doc_version += 1
        self._send_notification("textDocument/didChange", {
            "textDocument": {"uri": uri, "version": self._doc_version},
            "contentChanges": [{"text": new_content}],
        })
        self._current_file_content = new_content

    def apply_tactic(self, tactic: str) -> Dict[str, Any]:
        """Apply a tactic by replacing the last ``sorry`` with the tactic.

        Returns a dict with keys: success, goals, error.
        """
        if not self._current_file:
            return {"success": False, "goals": [], "error": "No file open"}

        content = self._current_file_content
        tactic_block = tactic.strip()
        idx = content.rfind("sorry")

        if idx >= 0:
            line_start = content.rfind("\n", 0, idx) + 1
            indent = content[line_start:idx]
            indented_tactic = ("\n" + indent).join(tactic_block.split("\n"))
            new_content = content[:idx] + indented_tactic + content[idx + 5:]
        else:
            new_content = content.rstrip() + "\n" + textwrap.indent(tactic_block, "  ")

        self.update_file(new_content)
        time.sleep(0.3)
        return self.get_goals()

    def get_goals(self) -> Dict[str, Any]:
        """Query the current goal state via ``$/lean/rpc/call`` ``getInteractiveGoals``."""
        if not self._started or not self._current_file:
            return {"success": False, "goals": [], "error": "No active file"}

        uri = self._path_to_uri(self._current_file)
        lines = self._current_file_content.split("\n")
        last_line = max(0, len(lines) - 1)
        last_char = len(lines[-1]) if lines else 0

        try:
            result = self._send_request("$/lean/rpc/call", {
                "uri": uri,
                "method": "getInteractiveGoals",
                "params": {"p": {"line": last_line, "character": last_char}},
            })
            return self._parse_goal_response(result)
        except (LSPError, LeanServerError) as e:
            return {"success": False, "goals": [], "error": str(e)}

    # ── Evaluating Expressions ───────────────────────────────────────────

    def eval_expr(self, expression: str) -> Dict[str, Any]:
        """Evaluate a Lean expression via ``#eval`` as a subprocess."""
        try:
            result = subprocess.run(
                [self.lean_path, "--eval", f"#eval {expression}"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self._workspace_dir,
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout.strip(),
                "error": result.stderr.strip() if result.stderr else None,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Evaluation timed out"}
        except (FileNotFoundError, OSError):
            return {"success": False, "error": f"Lean executable not found: {self.lean_path}"}

    # ── Internal LSP Methods ─────────────────────────────────────────────

    def _send_initialize(self) -> None:
        """Send LSP initialize request and store server capabilities."""
        result = self._send_request("initialize", {
            "processId": os.getpid() if hasattr(os, "getpid") else 0,
            "clientInfo": {"name": "axiom-zero", "version": "0.1.0"},
            "capabilities": {
                "textDocument": {
                    "synchronization": {"didOpen": True, "didChange": True},
                }
            },
        })
        self._server_capabilities = result.get("capabilities", {})
        self._lean_version = result.get("serverInfo", {}).get("version", "unknown")
        self._send_notification("initialized", {})
        self._log(f"Connected to Lean {self._lean_version}")

    def _send_request(self, method: str, params: Any = None) -> Dict[str, Any]:
        """Send a JSON-RPC 2.0 request and wait for the response."""
        if not self._process or not self._process.stdin:
            raise LeanServerError("Server not connected")

        self._request_id += 1
        msg: Dict[str, Any] = {"jsonrpc": "2.0", "id": self._request_id, "method": method}
        if params is not None:
            msg["params"] = params

        frame = _encode_lsp_message(msg)
        self._log(f"-> {method} (id={self._request_id})")

        try:
            self._process.stdin.write(frame)
            self._process.stdin.flush()
        except BrokenPipeError:
            raise LeanServerError("Server process died")

        return self._read_response(self._request_id)

    def _send_notification(self, method: str, params: Any = None) -> None:
        """Send a JSON-RPC 2.0 notification (no response expected)."""
        if not self._process or not self._process.stdin:
            return
        msg: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        try:
            self._process.stdin.write(_encode_lsp_message(msg))
            self._process.stdin.flush()
        except BrokenPipeError:
            pass

    def _start_reader_thread(self) -> None:
        """Start a daemon thread reading stdout from the Lean server.

        Using a background thread avoids ``select.select``, which does not work
        on Windows pipes.
        """
        if not self._process or not self._process.stdout:
            return

        def _reader():
            try:
                while self._started:
                    chunk = self._process.stdout.read1(65536)
                    if not chunk:
                        break
                    with self._rcv_lock:
                        self._rcv_buffer += chunk
                    self._rcv_event.set()
            except (ValueError, OSError):
                pass

        self._reader_thread = threading.Thread(target=_reader, daemon=True)
        self._reader_thread.start()

    def _read_response(self, expected_id: int) -> Dict[str, Any]:
        """Read and parse an LSP response from the server (thread-safe)."""
        if not self._process or not self._process.stdout:
            raise LeanServerError("Server not connected")

        start_time = time.monotonic()

        while time.monotonic() - start_time < self.timeout:
            for i, msg in enumerate(self._pending_messages):
                if msg.get("id") == expected_id:
                    self._pending_messages.pop(i)
                    if "error" in msg:
                        err = msg["error"]
                        raise LSPError(err.get("code", -1), err.get("message", ""), err.get("data"))
                    return msg.get("result", {})

            self._rcv_event.wait(timeout=0.3)
            self._rcv_event.clear()

            if self._process.poll() is not None:
                stderr = self._read_stderr() or ""
                raise LeanServerError(f"Server process died (exit code: {self._process.returncode}). Stderr: {stderr[:300]}")

            with self._rcv_lock:
                if not self._rcv_buffer:
                    continue
                messages, bytes_consumed = _parse_lsp_response(self._rcv_buffer)
                if bytes_consumed > 0:
                    self._rcv_buffer = self._rcv_buffer[bytes_consumed:]

            if not messages:
                continue

            for msg in messages:
                if msg.get("method") == "textDocument/publishDiagnostics":
                    self._log(f"(diagnostics) {json.dumps(msg.get('params', {}), ensure_ascii=False)[:200]}")
                    continue
                if "method" in msg and "id" not in msg:
                    self._log(f"(server notification: {msg['method']})")
                    continue
                if "id" in msg:
                    if msg["id"] != expected_id:
                        self._pending_messages.append(msg)
                        continue
                    if "error" in msg:
                        err = msg["error"]
                        raise LSPError(err.get("code", -1), err.get("message", ""), err.get("data"))
                    return msg.get("result", {})

        raise LeanServerError(f"Timeout waiting for LSP response (id={expected_id})")

    def _read_stderr(self) -> str:
        if not self._process or not self._process.stderr:
            return ""
        try:
            return self._process.stderr.read().decode("utf-8", errors="replace")
        except Exception:
            return ""

    # ── Response Parsing ─────────────────────────────────────────────────

    def _parse_goal_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the ``getInteractiveGoals`` RPC response into structured goals."""
        goals_data = response.get("goals", response.get("result", []))
        error = response.get("error")
        parsed_goals: List[Goal] = []

        if isinstance(goals_data, list):
            for i, g in enumerate(goals_data):
                goal_type = g.get("type", g.get("target", "?"))
                hyps_data = g.get("hyps", g.get("hypotheses", []))
                hypotheses = [
                    Hypothesis(
                        name=h.get("name", h.get("userName", "?")),
                        type=h.get("type", h.get("typeString", "?")),
                        is_parameter=h.get("isParameter", False),
                        is_inductive=h.get("isInductive", False),
                    )
                    for h in hyps_data if isinstance(h, dict)
                ]
                parsed_goals.append(Goal(
                    id=g.get("id", g.get("goal", str(uuid.uuid4()))),
                    type=goal_type,
                    hypotheses=hypotheses,
                    depth=i,
                ))

        return {
            "success": len(parsed_goals) > 0 or not error,
            "goals": parsed_goals,
            "goal_dicts": goals_data if isinstance(goals_data, list) else [],
            "error": error,
        }

    # ── Utilities ────────────────────────────────────────────────────────

    def _ensure_lean_available(self) -> None:
        """Verify Lean 4 is available, raising LeanServerError if not."""
        try:
            result = subprocess.run(
                [self.lean_path, "--version"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                raise LeanServerError(
                    f"Lean 4 not found at '{self.lean_path}'. "
                    "Install via: curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh"
                )
        except FileNotFoundError:
            raise LeanServerError(
                f"Lean 4 not found at '{self.lean_path}'. "
                "Install via: curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh"
            )

    def _path_to_uri(self, path: str) -> str:
        """Convert a file path to a file:// URI."""
        abs_path = os.path.abspath(path)
        if sys.platform == "win32":
            abs_path = abs_path.replace("\\", "/")
            if abs_path[1] == ":":
                abs_path = f"/{abs_path[0].lower()}:/{abs_path[3:]}"
        return f"file://{abs_path}"

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[LeanEnv] {message}", file=sys.stderr)

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._started and self._process is not None and self._process.poll() is None

    @property
    def lean_version(self) -> Optional[str]:
        return self._lean_version

    @property
    def current_file(self) -> Optional[str]:
        return self._current_file

    @property
    def workspace_dir(self) -> str:
        return self._workspace_dir
