"""
Axiom Zero - Lean 4 Environment Manager

Manages a Lean 4 server subprocess and communicates with it via JSON-RPC 2.0.
Handles server lifecycle (startup, shutdown), file management, tactic execution,
and goal state queries.

Communication protocol:
1. Spawn `lean --server` as a subprocess
2. Send JSON-RPC initialize handshake
3. Open a temporary .lean file via textDocument/didOpen
4. Execute tactics via mcp/lean_run_code or inline proof commands
5. Query goal states via mcp/lean_goal
6. Shutdown on completion

JSON-RPC 2.0 messages are sent as newline-delimited JSON over stdin/stdout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from .proof_state import (
    ProofState,
    Goal,
    Hypothesis,
    GoalStatus,
    TacticStep,
)


# ─── JSON-RPC Utilities ────────────────────────────────────────────────────

class JSONRPCError(Exception):
    """Error from a JSON-RPC response."""
    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"JSON-RPC error {code}: {message}")


class LeanServerError(Exception):
    """Error from the Lean 4 server."""
    pass


def make_request(method: str, params: Any = None, request_id: Optional[int] = None) -> Dict[str, Any]:
    """Create a JSON-RPC 2.0 request."""
    if request_id is None:
        request_id = int(time.time() * 1000) % 100000
    msg: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
    }
    if params is not None:
        msg["params"] = params
    return msg


def make_notification(method: str, params: Any = None) -> Dict[str, Any]:
    """Create a JSON-RPC 2.0 notification (no id)."""
    msg: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": method,
    }
    if params is not None:
        msg["params"] = params
    return msg


# ─── Lean 4 Environment ────────────────────────────────────────────────────

class LeanEnv:
    """
    Manages a Lean 4 server subprocess for proof automation.
    
    Usage:
        env = LeanEnv()
        env.start()
        
        # Execute a tactic
        result = env.run_tactic("intro x")
        
        # Get current goal state
        state = env.get_goal_state()
        
        # Evaluate a complete expression
        result = env.eval_expr("1 + 1")
        
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
        Initialize the Lean environment manager.
        
        Args:
            lean_path: Path to the Lean 4 executable
            workspace_dir: Directory for temporary .lean files
            timeout: Timeout in seconds for server responses
            verbose: Whether to print debug information
        """
        self.lean_path = lean_path
        self.timeout = timeout
        self.verbose = verbose

        # Server process
        self._process: Optional[subprocess.Popen] = None
        self._started: bool = False
        self._request_id: int = 0

        # Workspace management
        self._workspace_dir = workspace_dir or tempfile.mkdtemp(prefix="axiom_zero_lean_")
        self._current_file: Optional[str] = None
        self._lean_version: Optional[str] = None

        # Server capabilities
        self._server_capabilities: Dict[str, Any] = {}

        # Buffered output
        self._buffer: str = ""

    # ─── Lifecycle Management ────────────────────────────────────────────

    def start(self) -> bool:
        """
        Start the Lean 4 server.
        
        Returns:
            True if the server started successfully
            
        Raises:
            LeanServerError: If Lean 4 is not found or server fails to start
        """
        if self._started:
            self._log("Server already running")
            return True

        # Verify Lean 4 is available
        if not self._check_lean_available():
            raise LeanServerError(
                f"Lean 4 not found at '{self.lean_path}'. "
                "Install it via: curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh"
            )

        # Create workspace if needed
        Path(self._workspace_dir).mkdir(parents=True, exist_ok=True)

        try:
            self._process = subprocess.Popen(
                [self.lean_path, "--server"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self._workspace_dir,
                text=True,
                bufsize=1,  # Line buffered
            )
        except FileNotFoundError:
            raise LeanServerError(f"Could not execute Lean 4: {self.lean_path}")

        # Wait for server to start
        time.sleep(0.5)

        if self._process.poll() is not None:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise LeanServerError(
                f"Lean server exited immediately. Error: {stderr}"
            )

        self._started = True

        # Send initialize request
        self._initialize()

        self._log("Lean 4 server started successfully")
        return True

    def stop(self):
        """Shutdown the Lean 4 server gracefully."""
        if not self._started or not self._process:
            return

        try:
            # Send shutdown notification
            self._send_notification("shutdown")

            # Send exit notification
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
        self._log("Lean 4 server stopped")

    def restart(self) -> bool:
        """Restart the Lean 4 server."""
        self.stop()
        # Small delay to ensure clean restart
        time.sleep(0.3)
        return self.start()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ─── File Management ─────────────────────────────────────────────────

    def create_file(self, content: str, module_name: str = "Temp") -> str:
        """
        Create a temporary .lean file and open it in the server.
        
        Args:
            content: Lean 4 source code
            module_name: Name for the module
            
        Returns:
            Path to the created file
        """
        file_path = os.path.join(self._workspace_dir, f"{module_name}.lean")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        self._current_file = file_path

        # Open the file in the server
        self._send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": self._path_to_uri(file_path),
                "languageId": "lean4",
                "version": 1,
                "text": content,
            }
        })

        self._log(f"Created file: {file_path}")
        return file_path

    def create_temp_theorem(self, theorem_name: str, statement: str) -> str:
        """
        Create a temporary .lean file with a theorem to prove.
        
        Args:
            theorem_name: Name of the theorem
            statement: The theorem statement (e.g., "∀ (x : ℕ), x + 0 = x")
            
        Returns:
            Path to the temporary file
        """
        content = textwrap.dedent(f"""
        import Mathlib
        
        set_option pp.unicode.fun true
        
        theorem {theorem_name} : {statement} := by
          sorry
        """).strip()

        return self.create_file(content, theorem_name)

    # ─── Tactic Execution ─────────────────────────────────────────────────

    def run_tactic(self, tactic: str, goal_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute a tactic on the current proof state.
        
        Args:
            tactic: The tactic to execute (e.g., "intro x")
            goal_id: Optional specific goal to target
            
        Returns:
            Dict with keys: success, goals, error, output
            
        Raises:
            LeanServerError: If the server is not running
        """
        if not self._started or not self._current_file:
            raise LeanServerError("Server not started. Call start() first.")

        # Replace the sorry with our tactic
        with open(self._current_file, "r") as f:
            content = f.read()

        # Find the last occurrence of "sorry" and replace it with the tactic
        tactic_block = self._build_tactic_block(tactic)
        new_content = content.rsplit("sorry", 1)
        if len(new_content) == 2:
            new_content = new_content[0] + tactic_block + new_content[1]
        else:
            new_content = content + f"\n  {tactic}"

        # Write updated content
        with open(self._current_file, "w") as f:
            f.write(new_content)

        # Notify server of the change
        self._send_notification("textDocument/didChange", {
            "textDocument": {
                "uri": self._path_to_uri(self._current_file),
                "version": 2,
            },
            "contentChanges": [
                {"text": new_content}
            ]
        })

        # Wait a moment for the server to process
        time.sleep(0.2)

        # Get the updated goal state
        return self.get_goal_state()

    def run_code(self, code: str) -> Dict[str, Any]:
        """
        Run Lean code (e.g., a tactic block) and return the result.
        Uses mcp/lean_run_code if available, otherwise falls back to file-based approach.
        
        Args:
            code: Lean 4 code to execute
            
        Returns:
            Dict with keys: success, output, error
        """
        if not self._started:
            raise LeanServerError("Server not started. Call start() first.")

        # Try MCP method first
        if "mcp/lean_run_code" in self._server_capabilities.get("experimental", {}).get("mcpMethods", []):
            return self._send_request("mcp/lean_run_code", {
                "code": code,
            })
        else:
            # Fallback: create a temp file and evaluate
            return self._eval_via_file(code)

    def get_goal_state(self) -> Dict[str, Any]:
        """
        Query the current goal state from the Lean server.
        
        Returns:
            Dict with keys: success, goals (list of goal dicts), error
        """
        if not self._started or not self._current_file:
            return {"success": False, "goals": [], "error": "No active file"}

        try:
            result = self._send_request("mcp/lean_goal", {
                "uri": self._path_to_uri(self._current_file),
                "position": {"line": 9999, "character": 0},  # End of file
            })
            return self._parse_goal_response(result)
        except JSONRPCError as e:
            return {"success": False, "goals": [], "error": str(e)}

    def eval_expr(self, expression: str) -> Dict[str, Any]:
        """
        Evaluate a Lean 4 expression and return the result.
        
        Args:
            expression: Lean expression to evaluate
            
        Returns:
            Dict with keys: success, result, type, error
        """
        code = textwrap.dedent(f"""
        #eval {expression}
        """).strip()

        return self.run_code(code)

    # ─── Internal JSON-RPC Methods ───────────────────────────────────────

    def _initialize(self):
        """Send JSON-RPC initialize request."""
        params = {
            "processId": os.getpid(),
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

        self._log(f"Connected to Lean {self._lean_version}")

    def _send_request(self, method: str, params: Any = None) -> Dict[str, Any]:
        """
        Send a JSON-RPC request and wait for the response.
        
        Args:
            method: JSON-RPC method name
            params: Parameters for the method
            
        Returns:
            Response result
            
        Raises:
            JSONRPCError: If the server returns an error
            LeanServerError: If communication fails
        """
        if not self._process or not self._process.stdin:
            raise LeanServerError("Server not connected")

        self._request_id += 1
        request = make_request(method, params, self._request_id)
        request_str = json.dumps(request) + "\n"

        self._log(f"→ {method} (id={self._request_id})")

        try:
            self._process.stdin.write(request_str)
            self._process.stdin.flush()
        except BrokenPipeError:
            raise LeanServerError("Server process died")

        # Read response
        return self._read_response(self._request_id)

    def _send_notification(self, method: str, params: Any = None):
        """Send a JSON-RPC notification (no response expected)."""
        if not self._process or not self._process.stdin:
            return

        notification = make_notification(method, params)
        notif_str = json.dumps(notification) + "\n"

        self._log(f"→ (notify) {method}")

        try:
            self._process.stdin.write(notif_str)
            self._process.stdin.flush()
        except BrokenPipeError:
            pass

    def _read_response(self, expected_id: int) -> Dict[str, Any]:
        """
        Read a JSON-RPC response from the server.
        
        Args:
            expected_id: The expected request ID
            
        Returns:
            Response result dict
            
        Raises:
            JSONRPCError: If the server returns an error
        """
        if not self._process or not self._process.stdout:
            raise LeanServerError("Server not connected")

            start_time = time.time()
        accumulated = ""

        while time.time() - start_time < self.timeout:
            # Check if data is available using polling (Windows-compatible)

            line = self._process.stdout.readline()
            if not line:
                # Check if process died
                if self._process.poll() is not None:
                    raise LeanServerError(f"Server process died (exit code: {self._process.returncode})")
                continue

            accumulated += line
            line = line.strip()

            if not line:
                continue

            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                # Server might send non-JSON diagnostic messages
                self._log(f"(non-JSON) {line.strip()}")
                continue

            # Handle responses
            if "id" in response:
                resp_id = response["id"]
                if resp_id == expected_id:
                    if "error" in response:
                        err = response["error"]
                        raise JSONRPCError(err.get("code", -1), err.get("message", ""), err.get("data"))
                    return response.get("result", {})
                else:
                    # Response for a different request, store it
                    self._log(f"(unexpected id: {resp_id}, expected: {expected_id})")
                    continue

            # Handle notifications from server (no id)
            method = response.get("method", "")
            if method == "textDocument/publishDiagnostics":
                self._log(f"(diagnostics) {response}")

        raise LeanServerError(f"Timeout waiting for response (id={expected_id})")

    def _build_tactic_block(self, tactic: str) -> str:
        """Build a tactic block from a single tactic string."""
        return f"\n  {tactic}"

    def _parse_goal_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the goal state from a Lean server response.
        
        Args:
            response: Raw JSON-RPC response
            
        Returns:
            Structured dict with goals, success status
        """
        goals = response.get("goals", [])
        error = response.get("error")

        parsed_goals = []
        for i, goal_data in enumerate(goals):
            goal_id = goal_data.get("id", str(uuid.uuid4()))
            goal_type = goal_data.get("type", goal_data.get("target", "?"))
            hypotheses = []

            # Parse hypotheses from context
            for hyp_data in goal_data.get("hypotheses", []):
                hyp = Hypothesis(
                    name=hyp_data.get("name", "?"),
                    type=hyp_data.get("type", "?"),
                    is_parameter=hyp_data.get("isParameter", False),
                    is_inductive=hyp_data.get("isInductive", False),
                )
                hypotheses.append(hyp)

            goal = Goal(
                id=goal_id,
                type=goal_type,
                hypotheses=hypotheses,
                depth=i,
            )
            parsed_goals.append(goal)

        return {
            "success": len(parsed_goals) > 0 or not error,
            "goals": parsed_goals,
            "goal_dicts": goals,  # Keep the raw data too
            "error": error,
        }

    def _eval_via_file(self, code: str) -> Dict[str, Any]:
        """
        Evaluate code by writing it to a temporary file.
        Fallback when MCP methods are not available.
        """
        # Create a temp lean file with the code
        eval_file = os.path.join(self._workspace_dir, f"_eval_{int(time.time())}.lean")
        with open(eval_file, "w") as f:
            f.write(code)

        try:
            # Run lean --eval on it
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
        finally:
            if os.path.exists(eval_file):
                os.remove(eval_file)

    # ─── Utility Methods ─────────────────────────────────────────────────

    def _check_lean_available(self) -> bool:
        """Check if the Lean 4 executable is available."""
        try:
            result = subprocess.run(
                [self.lean_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _path_to_uri(self, path: str) -> str:
        """Convert a file path to a file:// URI."""
        abs_path = os.path.abspath(path)
        return f"file://{abs_path.replace(os.sep, '/')}"

    def _log(self, message: str):
        """Print debug information if verbose mode is enabled."""
        if self.verbose:
            print(f"[LeanEnv] {message}", file=sys.stderr)

    @property
    def is_running(self) -> bool:
        """Check if the server process is running."""
        return self._started and self._process is not None and self._process.poll() is None

    @property
    def lean_version(self) -> Optional[str]:
        """Get the Lean 4 version string."""
        return self._lean_version
