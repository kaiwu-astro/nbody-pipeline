"""
Shell command utilities
"""

from subprocess import run
from typing import List


def get_output(cmd: str, raise_error: bool = False) -> List[str]:
    """
    Execute a shell command and return its output.

    Args:
        cmd: Shell command to execute
        raise_error: If True, raise error on command failure (default: False)

    Returns:
        List of strings, one per line of output (without trailing newlines)
    """
    result = run(cmd, shell=True, check=raise_error, capture_output=True)
    return str(result.stdout, encoding="utf-8").split("\n")[:-1]
