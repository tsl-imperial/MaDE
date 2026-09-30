"""Checkpoint helpers with Orbax-first persistence and pickle fallbacks."""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import optax

try:
    import orbax.checkpoint as ocp
except Exception as exc:  # pragma: no cover - depends on installed JAX/Orbax pair.
    ocp = None
    _ORBAX_IMPORT_ERROR = exc
else:
    _ORBAX_IMPORT_ERROR = None

try:  # cloudpickle handles JAX custom_jvp activation objects stored in eqx.nn.MLP.
    import cloudpickle as pickle
except ImportError:  # pragma: no cover - standard pickle is the best available fallback.
    import pickle


class TrainState(eqx.Module):
    """Serializable training state shared by the trainer and scripts."""

    model: eqx.Module
    opt_state_I: optax.OptState
    opt_state_T: optax.OptState
    key: jax.Array
    step: int
    phase: int = 1


class CheckpointManager:
    """Wrapper around Orbax checkpoint directories used by the project."""

    def __init__(self, directory: str, max_to_keep: int = 3, save_interval: int = 1000):
        # Orbax tensorstore raises in a daemon thread when given a relative path, so the
        # main process can exit 0 on a silent partial save. Absolutise to avoid that.
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
        self._manager = None
        if ocp is not None:
            try:
                self._manager = ocp.CheckpointManager(
                    self.directory,
                    options=ocp.CheckpointManagerOptions(max_to_keep=max_to_keep),
                )
            except Exception:
                self._manager = None

    def _step_dir(self, step: int) -> Path:
        return self.directory / str(step)

    def save(self, state: TrainState, step: int) -> None:
        """Persist a train state at a given step."""
        payload = {
            "opt_state_I": state.opt_state_I,
            "opt_state_T": state.opt_state_T,
            "key": state.key,
            "step": state.step,
            "phase": state.phase,
        }

        if self._manager is not None and ocp is not None:
            try:
                args = ocp.args.StandardSave(payload)
                self._manager.save(step, args=args)
                if hasattr(self._manager, "wait_until_finished"):
                    self._manager.wait_until_finished()
            except Exception:
                pass

        # Orbax may create or populate the numeric step directory depending on the
        # installed Orbax/JAX pair. Write the pickle fallback after the Orbax attempt
        # (and after any async save finishes) so restore() has a stable location
        # regardless of Orbax API drift.
        step_dir = self._step_dir(step)
        step_dir.mkdir(parents=True, exist_ok=True)
        with step_dir.joinpath("state.pkl").open("wb") as handle:
            pickle.dump(state, handle)
        eqx.tree_serialise_leaves(step_dir / "model.eqx", state.model)
        with step_dir.joinpath("orbax_fallback.pkl").open("wb") as handle:
            pickle.dump(payload, handle)

    def restore(self, step: int | None = None) -> TrainState | None:
        """Restore the latest or requested train state."""
        target_step = self.latest_step() if step is None else step
        if target_step is None:
            return None

        step_dir = self._step_dir(target_step)
        pickle_path = step_dir / "state.pkl"
        if pickle_path.exists():
            with pickle_path.open("rb") as handle:
                return pickle.load(handle)

        try:
            if self._manager is None:
                raise RuntimeError("Orbax checkpoint manager is unavailable.")
            restored = self._manager.restore(target_step)
        except Exception:
            fallback_path = step_dir / "orbax_fallback.pkl"
            if not fallback_path.exists():
                return None
            with fallback_path.open("rb") as handle:
                restored = pickle.load(handle)

        model_path = step_dir / "model.eqx"
        model = restored.get("model")
        if model is None:
            raise ValueError(
                "Checkpoint restore requires a serialised model or a pickled TrainState fallback."
            )
        if model_path.exists():
            model = eqx.tree_deserialise_leaves(model_path, model)

        return TrainState(
            model=model,
            opt_state_I=restored["opt_state_I"],
            opt_state_T=restored["opt_state_T"],
            key=restored["key"],
            step=int(restored["step"]),
            phase=int(restored.get("phase", 1)),
        )

    def latest_step(self) -> int | None:
        """Return the latest checkpoint step if present."""
        candidates: list[int] = []
        if self._manager is not None:
            try:
                latest = self._manager.latest_step()
                if latest is not None:
                    candidates.append(int(latest))
            except Exception:
                pass

        candidates.extend(
            int(path.name)
            for path in self.directory.iterdir()
            if path.is_dir() and path.name.isdigit()
        )
        if not candidates:
            return None
        return max(candidates)
