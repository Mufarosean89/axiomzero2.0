"""
Axiom Zero - Lean 4 Environment Manager

Manages a Lean 4 server subprocess and communicates with it via the Language
Server Protocol (LSP) over JSON-RPC 2.0.

Communication protocol (LSP over stdio):
----------------------------------------
1. Spawn `lean --server` as a subprocess (binary pipes).
2. Send `Content-Length: N\r\n\r\n{json}` frames to stdin.
3. Read `Content-Length: N\r\n\r\n{json}` frames from stdout.
4. Standard LSP handshake: initialize → textDocument/didOpen → ...
5. Lean-specific RPC: `$/lean/rpc/call` with `getInteractiveGoals` to query
   the goal state at a given cursor position.
6. File changes via `textDocument/didChange`.
7. Shutdown on completion.

Usage:
    env = LeanEnv()
    env.start()

    # Open a file
    env.open_file("theorem add_comm : ... := by\\n  sorry\\n")

    # Apply a tactic by replacing the last "sorry" and updating the file
    env.apply_tactic("intro x y")

    # Get goal state
    goals = env.get_goals()

    env.stop()
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .proof_state import (
    ProofState,
    Goal,
    Hypothesis,
    GoalStatus,
    TacticStep,
)


# ─── LSP Protocol Helpers ────────────────────────────────────────────────────

def _encode_lsp_message(msg: Dict[str, Any]) -> bytes:
    """Encode a JSON-RPC message as an LSP frame with Content-Length header.

    Format::
        Content-Length: N\r\n\r\n{json}
    """
    body = json.dumps(msg, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def _parse_lsp_response(data: bytes) -> Tuple[List[Dict[str, Any]], int]:
    """Parse LSP frames from a byte buffer.

    Returns:
        (messages, bytes_consumed)
        messages: List of parsed JSON-RPC messages.
        bytes_consumed: Total number of bytes consumed from the buffer.
    """
    messages: List[Dict[str, Any]] = []
    bytes_consumed = 0
    remaining = data

    while remaining:
        # Try to find a Content-Length header
        header_match = re.match(
            rb"Content-Length:\s*(\d+)\s*\r?\n\r?\n",
            remaining,
            re.IGNORECASE,
        )
        if not header_match:
            # No valid header found at the start — stop
            break

        content_length = int(header_match.group(1))
        header_end = header_match.end()
        body_start = header_end
        body_end = body_start + content_length

        if body_end > len(remaining):
            # Incomplete message — wait for more data
            break

        body = remaining[body_start:body_end]
        try:
            msg = json.loads(body)
            messages.append(msg)
        except json.JSONDecodeError:
            pass  # Skip malformed messages

        frame_len = body_end
        bytes_consumed += frame_len
        remaining = remaining[frame_len:]

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


# ─── JSON-RPC 2.0 Message Helpers ────────────────────────────────────────────

_request_counter = 0


def make_request(method: str, params: Any = None) -> Dict[str, Any]:
    """Create a JSON-RPC 2.0 request dict.

    Args:
        method: Method name.
        params: Parameters.

    Returns:
        A JSON-RPC 2.0 request dict with an auto-incremented id.
    """
    global _request_counter
    _request_counter += 1
    msg: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": _request_counter,
        "method": method,
    }
    if params is not None:
        msg["params"] = params
    return msg


def make_notification(method: str, params: Any = None) -> Dict[str, Any]:
    """Create a JSON-RPC 2.0 notification dict (no id field).

    Args:
        method: Method name.
        params: Parameters.

    Returns:
        A JSON-RPC 2.0 notification dict.
    """
    msg: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": method,
    }
    if params is not None:
        msg["params"] = params
    return msg


# ─── Lean 4 Environment ──────────────────────────────────────────────────────

class LeanEnv:
    """
    Manages a Lean 4 server subprocess using LSP over stdio.

    Usage:
        env = LeanEnv()
        env.start()

        # Create a theorem file
        env.open_file("theorem hello : 1 + 1 = 2 := by\\n  sorry\\n")

        # Apply a tactic (replaces the last ``sorry``)
        env.apply_tactic("rfl")

        # Check goals
        result = env.get_goals()
        if result["success"]:
            print(f"Goals: {result['goals']}")

        env.stop()
    """

    def __init__(
        self,
        lean_path: str = "lean",
        workspace_dir: Optional[str] = None,
        timeout: float = 30.0,
        verbose: bool = False,
    ):
        """
        Args:
            lean_path: Path to the Lean 4 executable.
            workspace_dir: Directory for temporary .lean files.
            timeout: Timeout in seconds for server responses.
            verbose: Print debug information.
        """
        self.lean_path = lean_path
        self.timeout = timeout
        self.verbose = verbose

        # Server process (binary pipes — critical for Windows compatibility)
        self._process: Optional[subprocess.Popen] = None
        self._started: bool = False
        self._request_id: int = 0
        self._rcv_buffer = b""  # Buffer for partially-read LSP frames (thread-safe)
        self._rcv_lock = threading.Lock()  # Guards _rcv_buffer
        self._rcv_event = threading.Event()  # Signals new data available
        self._reader_thread: Optional[threading.Thread] = None

        # Workspace / file management
        self._workspace_dir = workspace_dir or tempfile.mkdtemp(prefix="axiom_zero_lean_")
        self._current_file: Optional[str] = None
        self._current_file_content: str = ""
        self._lean_version: Optional[str] = None
        self._server_capabilities: Dict[str, Any] = {}

        # Pending RPC responses that arrived before their consumer was ready
        self._pending_messages: List[Dict[str, Any]] = []

        # Thread for reading server notifications (diagnostics, etc.)
        self._notification_thread = None

    # ─── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Start the Lean 4 server and perform LSP initialize handshake.

        Returns:
            True if successful.

        Raises:
            LeanServerError: If Lean is not found or the server fails.
        """
        if self._started:
            self._log("Server already running")
            return True

        # Verify Lean 4 is available
        if not self._check_lean_available():
            raise LeanServerError(
                f"Lean 4 not found at '{self.lean_path}'. "
                "Install via: curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh"
            )

        # Create workspace
        Path(self._workspace_dir).mkdir(parents=True, exist_ok=True)

        try:
            # IMPORTANT: binary pipes (not text mode) to avoid Windows \\r\\n mangling
            self._process = subprocess.Popen(
                [self.lean_path, "--server"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self._workspace_dir,
            )
        except FileNotFoundError:
            raise LeanServerError(f"Could not execute Lean 4: {self.lean_path}")

        # Give the server a moment to start
        time.sleep(0.5)

        if self._process.poll() is not None:
            stderr = self._read_stderr() or ""
            raise LeanServerError(
                f"Lean server exited immediately. Error: {stderr[:500]}"
            )

        self._started = True

        # Start the background reader thread (avoids select.select on Windows pipes)
        self._start_reader_thread()

        # LSP initialize handshake
        self._initialize()

        self._log("Lean 4 server started successfully")
        return True

    def stop(self) -> None:
        """Shut down the Lean 4 server gracefully."""
        if not self._started or not self._process:
            return

        try:
            # Shutdown notification
            self._send_notification("shutdown")
            # Exit notification
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

        # Signal the reader thread to stop
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

    # ─── File Management ──────────────────────────────────────────────────

    def open_file(self, content: str, module_name: str = "Temp") -> str:
        """Create a .lean file and open it on the server via textDocument/didOpen.

        Args:
            content: Lean 4 source code.
            module_name: Module/file name (without .lean extension).

        Returns:
            Path to the created file.
        """
        file_path = os.path.join(self._workspace_dir, f"{module_name}.lean")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        self._current_file = file_path
        self._current_file_content = content

        # Notify server
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
        """Update the current file on the server via textDocument/didChange.

        Args:
            new_content: The full new file content.
        """
        if not self._current_file:
            raise LeanServerError("No file open. Call open_file() first.")

        uri = self._path_to_uri(self._current_file)

        self._send_notification("textDocument/didChange", {
            "textDocument": {
                "uri": uri,
                "version": self._get_next_version(),
            },
            "contentChanges": [
                {"text": new_content}
            ],
        })

        self._current_file_content = new_content
        self._log("File updated")

    def apply_tactic(self, tactic: str) -> Dict[str, Any]:
        """Apply a tactic by replacing the last ``sorry`` with the tactic.

        This is the primary way to advance the proof state. The tactic is
        appended in place of the last ``sorry``, and the file is updated
        via textDocument/didChange.

        Args:
            tactic: The Lean 4 tactic string (e.g., ``intro x``).

        Returns:
            Dict with keys: success, goals, error
        """
        if not self._current_file:
            return {"success": False, "goals": [], "error": "No file open"}

        content = self._current_file_content

        # Find and replace the last ``sorry`` with the tactic block
        # This handles both ``sorry`` on its own line and inline
        tactic_block = tactic.strip()
        # Build the indentation-aware replacement
        idx = content.rfind("sorry")
        if idx >= 0:
            # Determine indentation from the line where "sorry" appears
            line_start = content.rfind("\n", 0, idx) + 1
            indent = content[line_start:idx]
            # For multi-line tactics, preserve indentation
            indented_tactic = ("\n" + indent).join(tactic_block.split("\n"))
            new_content = content[:idx] + indented_tactic + content[idx + 5:]
        else:
            # No sorry found — just append the tactic on a new line
            new_content = content.rstrip() + "\n" + textwrap.indent(tactic_block, "  ")

        self.update_file(new_content)

        # Wait briefly for the server to process
        time.sleep(0.3)

        # Query the goal state at the end of the file
        return self.get_goals()

    def get_goals(self) -> Dict[str, Any]:
        """Query the current goal state via ``$/lean/rpc/call`` ``getInteractiveGoals``.

        Returns:
            Dict with keys:
                success    : bool
                goals      : list of Goal objects
                goal_dicts : list of raw goal dicts from the server
                error      : optional error string
        """
        if not self._started or not self._current_file:
            return {"success": False, "goals": [], "error": "No active file"}

        # Build the position at the end of the file (cursor = last line)
        uri = self._path_to_uri(self._current_file)
        lines = self._current_file_content.split("\n")
        last_line = max(0, len(lines) - 1)
        last_char = len(lines[-1]) if lines else 0

        try:
            result = self._send_request("$/lean/rpc/call", {
                "uri": uri,
                "method": "getInteractiveGoals",
                "params": {
                    "p": {
                        "line": last_line,
                        "character": last_char,
                    }
                },
            })
            return self._parse_goal_response(result)
        except (LSPError, LeanServerError) as e:
            return {"success": False, "goals": [], "error": str(e)}

    # ─── Evaluating Expressions ───────────────────────────────────────────

    def eval_expr(self, expression: str) -> Dict[str, Any]:
        """Evaluate a Lean expression via ``#eval``.

        Creates a temporary file, runs ``#eval <expr>``, and returns the
        result by running ``lean --eval`` as a subprocess.

        Args:
            expression: Lean expression to evaluate.

        Returns:
            Dict with keys: success, output, error
        """
        code = f'#eval {expression}'

        try:
            result = subprocess.run(
                [self.lean_path, "--eval", code],
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

    # ─── Internal LSP Methods ─────────────────────────────────────────────

    def _initialize(self) -> None:
        """Send LSP initialize request and store server capabilities."""
        params = {
            "processId": os.getpid() if hasattr(os, "getpid") else 0,
            "clientInfo": {
                "name": "axiom-zero",
                "version": "0.1.0",
            },
            "capabilities": {
                "textDocument": {
                    "synchronization": {
                        "didOpen": True,
                        "didChange": True,
                    }
                }
            },
        }

        result = self._send_request("initialize", params)
        self._server_capabilities = result.get("capabilities", {})
        self._lean_version = result.get("serverInfo", {}).get("version", "unknown")

        # Send initialized notification (required by LSP spec)
        self._send_notification("initialized", {})

        self._log(f"Connected to Lean {self._lean_version}")

    def _send_request(self, method: str, params: Any = None) -> Dict[str, Any]:
        """Send a JSON-RPC 2.0 request and wait for the response.

        Args:
            method: Method name.
            params: Parameters.

        Returns:
            Response result dict.

        Raises:
            LSPError: If the server returns an error.
            LeanServerError: If communication fails.
        """
        if not self._process or not self._process.stdin:
            raise LeanServerError("Server not connected")

        self._request_id += 1
        msg: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
        }
        if params is not None:
            msg["params"] = params

        frame = _encode_lsp_message(msg)
        self._log(f"→ {method} (id={self._request_id})")

        try:
            self._process.stdin.write(frame)
            self._process.stdin.flush()
        except BrokenPipeError:
            raise LeanServerError("Server process died")

        # Read response
        return self._read_response(self._request_id)

    def _send_notification(self, method: str, params: Any = None) -> None:
        """Send a JSON-RPC 2.0 notification (no response expected)."""
        if not self._process or not self._process.stdin:
            return

        msg: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            msg["params"] = params

        frame = _encode_lsp_message(msg)
        self._log(f"→ (notify) {method}")

        try:
            self._process.stdin.write(frame)
            self._process.stdin.flush()
        except BrokenPipeError:
            pass

    def _start_reader_thread(self) -> None:
        """Start a daemon thread that continuously reads stdout from the Lean server.

        Using a background thread avoids ``select.select``, which does not work
        on Windows pipes (raises ``OSError: WSAStartup``). The thread reads
        raw bytes and appends them to ``_rcv_buffer`` under a lock, then signals
        ``_rcv_event`` so that ``_read_response`` can wake up and parse.
        """
        if not self._process or not self._process.stdout:
            return

        def _reader():
            try:
                while self._started:
                    # Read available data (may return less than requested)
                    chunk = self._process.stdout.read1(65536)  # type: ignore
                    if not chunk:
                        break  # EOF — server closed stdout
                    with self._rcv_lock:
                        self._rcv_buffer += chunk
                    self._rcv_event.set()
            except (ValueError, OSError):
                pass  # Process terminated while reading

        self._reader_thread = threading.Thread(target=_reader, daemon=True)
        self._reader_thread.start()

    def _read_response(self, expected_id: int) -> Dict[str, Any]:
        """Read and parse an LSP response from the server (thread-safe).

        Args:
            expected_id: The expected request ID.

        Returns:
            Response result dict.

        Raises:
            LSPError: On error response.
            LeanServerError: On timeout or connection issues.
        """
        if not self._process or not self._process.stdout:
            raise LeanServerError("Server not connected")

        start_time = time.monotonic()

        while time.monotonic() - start_time < self.timeout:
            # First check pending messages from previous reads
            pending_idx = None
            for i, msg in enumerate(self._pending_messages):
                if msg.get("id") == expected_id:
                    pending_idx = i
                    break
            if pending_idx is not None:
                msg = self._pending_messages.pop(pending_idx)
                if "error" in msg:
                    err = msg["error"]
                    raise LSPError(
                        err.get("code", -1),
                        err.get("message", ""),
                        err.get("data"),
                    )
                return msg.get("result", {})

            # Wait for the reader thread to signal new data (or timeout)
            self._rcv_event.wait(timeout=0.3)
            self._rcv_event.clear()

            # Check if process died
            if self._process.poll() is not None:
                raise LeanServerError(
                    f"Server process died (exit code: {self._process.returncode})"
                )

            # Parse whatever is in the buffer under lock
            with self._rcv_lock:
                if not self._rcv_buffer:
                    continue
                messages, bytes_consumed = _parse_lsp_response(self._rcv_buffer)
                if bytes_consumed > 0:
                    self._rcv_buffer = self._rcv_buffer[bytes_consumed:]

            if not messages:
                continue

            # Process parsed messages
            for msg in messages:
                # Handle diagnostics notifications (no id)
                if msg.get("method") == "textDocument/publishDiagnostics":
                    self._log(
                        f"(diagnostics) "
                        f"{json.dumps(msg.get('params', {}), ensure_ascii=False)[:200]}"
                    )
                    continue

                # Handle notifications without id
                if "method" in msg and "id" not in msg:
                    self._log(f"(server notification: {msg['method']})")
                    continue

                # Handle responses with IDs
                if "id" in msg:
                    resp_id = msg["id"]
                    if resp_id != expected_id:
                        # Stash this response for the consumer that needs it
                        self._log(f"(stashing response id={resp_id}, expected={expected_id})")
                        self._pending_messages.append(msg)
                        continue

                    if "error" in msg:
                        err = msg["error"]
                        raise LSPError(
                            err.get("code", -1),
                            err.get("message", ""),
                            err.get("data"),
                        )
                    return msg.get("result", {})

        raise LeanServerError(f"Timeout waiting for LSP response (id={expected_id})")

    def _read_stderr(self) -> str:
        """Read all available stderr data."""
        if not self._process or not self._process.stderr:
            return ""
        try:
            return self._process.stderr.read().decode("utf-8", errors="replace")
        except Exception:
            return ""

    # ─── Response Parsing ─────────────────────────────────────────────────

    def _parse_goal_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the ``getInteractiveGoals`` RPC response into structured goals.

        Args:
            response: Raw result from ``$/lean/rpc/call``.

        Returns:
            Dict with keys: success, goals (list of Goal objects), goal_dicts, error
        """
        # The response structure depends on Lean's RPC implementation.
        # Common structures:
        #   {"goals": [{"hyps": [...], "type": "..."}, ...]}
        #   {"goals": [...], "error": "..."}

        goals_data = response.get("goals", response.get("result", []))
        error = response.get("error")

        parsed_goals: List[Goal] = []

        if isinstance(goals_data, list):
            for i, g in enumerate(goals_data):
                goal_type = g.get("type", g.get("target", "?"))
                hyps_data = g.get("hyps", g.get("hypotheses", []))

                hypotheses: List[Hypothesis] = []
                for h in hyps_data:
                    if isinstance(h, dict):
                        hypotheses.append(Hypothesis(
                            name=h.get("name", h.get("userName", "?")),
                            type=h.get("type", h.get("typeString", "?")),
                            is_parameter=h.get("isParameter", False),
                            is_inductive=h.get("isInductive", False),
                        ))

                goal = Goal(
                    id=g.get("id", g.get("goal", str(uuid.uuid4()))),
                    type=goal_type,
                    hypotheses=hypotheses,
                    depth=i,
                )
                parsed_goals.append(goal)

        return {
            "success": len(parsed_goals) > 0 or not error,
            "goals": parsed_goals,
            "goal_dicts": goals_data if isinstance(goals_data, list) else [],
            "error": error,
        }

    # ─── Utilities ────────────────────────────────────────────────────────

    def _check_lean_available(self) -> bool:
        """Check if the Lean 4 executable is available."""
        try:
            result = subprocess.run(
                [self.lean_path, "--version"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _path_to_uri(self, path: str) -> str:
        """Convert a file path to a file:// URI."""
        abs_path = os.path.abspath(path)
        # On Windows, convert backslashes and add leading slash for drive letter
        if sys.platform == "win32":
            abs_path = abs_path.replace("\\", "/")
            if abs_path[1] == ":":
                abs_path = f"/{abs_path[0].lower()}:/{abs_path[3:]}"
        return f"file://{abs_path}"

    def _get_next_version(self) -> int:
        """Increment and return the document version number."""
        if not hasattr(self, "_doc_version"):
            self._doc_version = 0
        self._doc_version += 1
        return self._doc_version

    def _log(self, message: str) -> None:
        """Print debug information if verbose mode is enabled."""
        if self.verbose:
            print(f"[LeanEnv] {message}", file=sys.stderr)

    # ─── Properties ───────────────────────────────────────────────────────

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
