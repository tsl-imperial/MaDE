# E1 metrics snapshot

`e1_metrics_snapshot.json` is a bitwise-frozen snapshot of every metric
helper that Experiment 1 paper tables depend on:

- `fidelity`
- `dynamics_violation_known`
- `dynamics_violation_learned`
- `dynamics_violation_true`
- `compute_metrics`

It guards against silent drift introduced when the real-data metrics for
Experiment 2 / Experiment 3 are added as siblings in
`src/made/evaluation/metrics.py` (ralplan-real-data-metrics-v1, R14 / Phase 0).

The test in `tests/test_e1_metrics_unchanged.py` evaluates the helpers on a
fixed input suite (standard batched, unbatched, length-1, empty, and a
near-π/2 steering edge case), then compares each value to this snapshot
using **bitwise float equality** (`==`).

## When the test fails

If a diff that does not touch `metrics.py` fails the snapshot, that is a
bug — investigate the actual cause (XLA flag drift, JAX upgrade, dtype
regression, etc.) before regenerating.

If a diff intentionally changes E1 metric definitions (rare; should
require explicit ADR sign-off), regenerate by:

```bash
MADE_E1_SNAPSHOT_REFRESH=1 JAX_PLATFORMS=cpu pytest tests/test_e1_metrics_unchanged.py -x
```

then commit the updated snapshot in the same change that motivates it.

## Exit ramp from bitwise → allclose

If the snapshot ever fails on a JAX upgrade where the underlying float64
arithmetic is provably equivalent (XLA reassociation, etc.) but not
bitwise identical, downgrade `_bitwise_equal` in
`tests/test_e1_metrics_unchanged.py` to
`numpy.testing.assert_allclose(rtol=1e-15, atol=1e-15)` and document the
JAX/XLA version pair in this file alongside a justification (cite the
upstream change). Do not weaken the tolerance further.
