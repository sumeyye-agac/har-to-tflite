from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class CommandResult:
    code: int
    stdout: str
    stderr: str


def run_command(cmd: list[str], timeout: float | None = None) -> CommandResult:
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    return CommandResult(code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
