"""Exclusive-lock guard against two processes writing the same output path.

`claim_output(path)` takes a lock beside the target and refuses if another live process holds
it, so a second run fails loudly at the start instead of silently overwriting a finished one.
Also records who holds the lock.

Usage::

    with claim_output(out_path):
        ...                       # compute
        out_path.write_text(...)  # write
"""

from __future__ import annotations

import errno
import json
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path


class OutputBusy(RuntimeError):
    """Raised when another live process has claimed the same output path."""


def _holder(lock: Path) -> dict:
    try:
        return json.loads(lock.read_text())
    except Exception:
        return {}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


@contextmanager
def claim_output(path: str | Path, *, force: bool = False):
    """Exclusively claim *path* for the duration of the block."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_suffix(target.suffix + ".lock")

    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            held = _holder(lock)
            pid = int(held.get("pid", -1))
            if pid > 0 and _alive(pid) and not force:
                raise OutputBusy(
                    f"{target} is claimed by pid {pid} on {held.get('host')} since "
                    f"{held.get('started')} ({held.get('cmd', '?')}). Refusing to run a second "
                    f"writer. Use a different --out, wait, or pass force=True if that process "
                    f"is known dead."
                )
            # Stale lock from a process that is gone: take it over.
            lock.unlink(missing_ok=True)

    try:
        os.write(
            fd,
            json.dumps(
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "cmd": " ".join(os.sys.argv[:3]),
                    "target": str(target),
                },
                indent=2,
            ).encode(),
        )
    finally:
        os.close(fd)

    try:
        yield target
    finally:
        lock.unlink(missing_ok=True)


def _visible_device_indices() -> tuple[int, ...] | None:
    """Physical indices this process can see, or None when it can see all of them."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or raw.strip() == "":
        return None
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return tuple(out) if out else None


def _bus_id_to_index() -> dict[str, int]:
    import subprocess

    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=30,
    ).stdout
    mapping: dict[str, int] = {}
    for line in out.strip().splitlines():
        if "," not in line:
            continue
        idx, bus = line.split(",", 1)
        if idx.strip().isdigit():
            mapping[bus.strip()] = int(idx.strip())
    return mapping


def require_idle_devices(
    *, allow_pids: tuple[int, ...] = (), devices: tuple[int, ...] | None = None
) -> None:
    """Refuse to start if another compute process holds a GPU.

    A latency harness sharing a device with another process measures contention, not the
    method: an overlapping job can inflate a measured median several times and its spread by
    an order of magnitude. Reads device state directly rather than tracking processes, and
    refuses rather than warns.

    `devices` scopes the check to those physical indices. Pass it only when the caller
    genuinely runs on a subset — a run pinned to one device by `CUDA_VISIBLE_DEVICES` is not
    slowed by a job on a different card. A latency measurement must leave it None (default,
    every device), since the blanket check is what catches contention.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout
        index_of = _bus_id_to_index() if devices is not None else {}
    except Exception as exc:  # pragma: no cover - no nvidia-smi is not a reason to proceed
        raise RuntimeError(
            f"cannot establish whether the devices are idle ({exc!r}); refusing to measure "
            f"latency against an unknown machine state"
        ) from exc

    mine = {os.getpid(), *allow_pids}
    busy = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        fields = [f.strip() for f in line.split(",")]
        pid_text = fields[0]
        if not pid_text.isdigit():
            continue
        pid = int(pid_text)
        if pid in mine:
            continue
        if devices is not None:
            bus = fields[1] if len(fields) > 1 else ""
            idx = index_of.get(bus)
            # An unmappable bus id is treated as contending: the guard must not pass because
            # it could not tell.
            if idx is not None and idx not in devices:
                continue
        busy.append(line.strip())

    if busy:
        raise RuntimeError(
            "REFUSING TO START: another compute process holds a GPU"
            + (f" this run would use ({devices})" if devices is not None else "")
            + ", so this would measure contention rather than the method.\n  "
            + "\n  ".join(busy)
            + "\nWait for the device to go idle, or pass its pid in allow_pids if it is "
              "known not to contend."
        )
