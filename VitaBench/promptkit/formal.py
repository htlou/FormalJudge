"""Explicit local Dafny verification of a supplied file; never runs agent tools."""
from pathlib import Path
import subprocess


def verify(path: Path, executable: str = "dafny", timeout: int = 60) -> dict:
    if path.suffix != ".dfy" or not path.is_file():
        raise ValueError("Supply an existing .dfy file")
    if not 1 <= timeout <= 600:
        raise ValueError("Timeout must be between 1 and 600 seconds")
    try:
        result = subprocess.run([executable, "verify", str(path.resolve())],
                                capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Dafny timed out; no verification record produced") from None
    except OSError:
        raise RuntimeError("Dafny could not start; check --dafny") from None
    # Dafny 4 exit-code contract: 0 success, 4 verification errors;
    # command-line, parsing/resolution and compilation errors are separate.
    # https://dafny.org/v4.3.0/DafnyRef/DafnyRef#1353-dafny-exit-codes
    error_type = None
    if result.returncode not in (0, 4):
        error_type = "execution_error"
    elif result.returncode == 4 and "timed out" in (result.stdout + result.stderr).lower():
        error_type = "timeout"
    return {
        "origin": "local_dafny_process",
        "status": "VERIFIED" if result.returncode == 0 else "UNPROVEN",
        "returncode": result.returncode,
        "error_type": error_type,
        "reasoning": "Dafny accepted the supplied specification." if result.returncode == 0 else "Dafny did not establish the supplied specification; inspect solver output.",
        "violations": [],
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
