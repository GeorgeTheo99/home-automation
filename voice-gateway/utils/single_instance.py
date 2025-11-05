"""Single-instance process lock using an advisory file lock (Linux/Unix).

This prevents multiple concurrent gateway processes from running and opening
duplicate audio streams. It uses fcntl.flock with an exclusive non-blocking
lock on a PID file. The file contains the active process PID for diagnostics.
"""

from __future__ import annotations

import fcntl
import os
from dataclasses import dataclass
import time
import signal
import logging
from pathlib import Path


class SingleInstanceError(RuntimeError):
    pass


@dataclass
class InstanceLock:
    path: Path
    fd: int

    def release(self) -> None:
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            os.close(self.fd)
        except Exception:
            pass


def _read_pid_from_fd(fd: int) -> int | None:
    try:
        with os.fdopen(fd, "r", closefd=False) as f:
            content = (f.read() or "").strip()
    except Exception:
        return None
    try:
        return int(content)
    except Exception:
        return None


def _proc_state(pid: int) -> str:
    """Return /proc state character for pid (e.g., 'R', 'S', 'T', 'Z')."""
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("State:"):
                    # Format: State:\tT (stopped) ...
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1]
    except Exception:
        pass
    return "?"


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Assume alive but inaccessible; better to treat as alive
        return True


def obtain_lock(path: Path) -> InstanceLock:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # Someone else holds the lock; see if it's a stopped/stale gateway and try to terminate it.
        other_pid = _read_pid_from_fd(fd)
        if other_pid is not None and _pid_is_alive(other_pid):
            state = _proc_state(other_pid)
            if state.upper().startswith("T"):
                # Process is stopped/suspended; try to terminate it nicely, then force.
                try:
                    logging.info(
                        "Instance lock held by stopped process (pid=%s); sending SIGTERM to release lock.",
                        other_pid,
                    )
                except Exception:
                    pass
                try:
                    os.kill(other_pid, signal.SIGTERM)
                except Exception:
                    pass
                # Wait briefly for lock release
                deadline = time.time() + 3.0
                while time.time() < deadline:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        time.sleep(0.1)
                else:
                    # Escalate to SIGKILL and one last attempt
                    try:
                        logging.warning(
                            "Previous instance (pid=%s) did not terminate; sending SIGKILL.", other_pid
                        )
                    except Exception:
                        pass
                    try:
                        os.kill(other_pid, signal.SIGKILL)
                    except Exception:
                        pass
                    time.sleep(0.2)
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        raise SingleInstanceError(
                            f"Another voice-gateway instance appears to be running (pid={other_pid})."
                        )
            else:
                raise SingleInstanceError(
                    f"Another voice-gateway instance appears to be running (pid={other_pid})."
                )
        else:
            # Unknown or dead holder; try one more time after a short wait.
            time.sleep(0.2)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SingleInstanceError(
                    f"Another voice-gateway instance appears to be running (pid={other_pid or 'unknown'})."
                )

    # Write our PID and truncate the file.
    try:
        with os.fdopen(fd, "w", closefd=False) as f:
            f.write(str(os.getpid()))
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass

    return InstanceLock(path=path, fd=fd)
