"""A refusal instead of a silent overwrite.

Two concurrent runs wrote one artifact today and the second overwrote the first five minutes
after it had been analysed and committed. The numbers happened to agree to 4.3e-16; that was
luck, not a property of the setup.

`claim_output(path)` takes an exclusive lock beside the target and refuses if another live
process holds it, so the second run **fails loudly at the start** rather than finishing and
overwriting. It also records who holds it, which is what was missing when the collision
surfaced as a `git status` line.

Usage::

    with claim_output(out_path):
        ...                       # compute
        out_path.write_text(...)  # write

Nothing here is specific to one script; every harness that writes a named artifact should use
it.
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

    A latency harness that runs while another process holds the device measures contention
    rather than the method: an overlapping job on the same device can inflate a measured
    median by several times and its spread by an order of magnitude.

    **The waiter that existed to prevent this is what let it through.** It captured only the
    FIRST matching process with `head -1` and released as soon as that one exited, while a
    second run of the same command was still going. A guard that can pass while its condition
    is false is the same class of defect as a test that cannot fail, so this one reads the
    device directly rather than tracking processes it was told about.

    **Refuses rather than warns.** A warning in a log is a thing nobody reads at the moment it
    matters.

    `devices` scopes the check to those PHYSICAL device indices. Pass it only when the caller
    genuinely runs on a subset — a run pinned to one device by `CUDA_VISIBLE_DEVICES` is not
    made slower by a job on a different card, and refusing on that would stop work for no
    reason on a two-card machine. **A latency measurement must not pass it**: leave it None
    there, because the blanket check is the one that catches this failure mode. `None` means
    every device, which stays the default.
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
