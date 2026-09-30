"""FAB phase 2 — latent space structuring, implemented from arXiv:2604.03489.

Implements the paper's second training phase; `fab_baseline` covers phase 1 plus the
latent projection only.

**What the paper specifies, and where.** Section 3.1, "Two-phase autoencoder training", and
Algorithm 1. Phase 1 is the L2 reconstruction of Eq. 4 on exclusively feasible points, which is
what `fab_baseline.train_fab_baseline` already does. Phase 2 introduces a discriminator and
trains the autoencoder on the composite `L_struc` of Eq. 6:

    L_struc = lambda_recon L_recon + lambda_hinge L_hinge
            + lambda_latent L_latent + lambda_geom L_geom                    (Eq. 6)

    L_disc(y, c)  = -(c log D(y) + (1-c) log(1 - D(y)))                      (Eq. 5)
    L_hinge(y, c) = c ReLU(||E(y)||_2 - r) + (1-c) ReLU(r - ||E(y)||_2)      (Eq. 7)
    L_latent(z)   = -log D(R(z)),           z ~ S                            (Eq. 8)
    L_geom(S_hat) = Var_{z~S}[log det(J_z J_z^T + eps I_k)]                  (Eq. 9)

The discriminator and the autoencoder are trained with *distinct* objectives, not a minimax
loss — the paper says so explicitly, and Algorithm 1 alternates them within each phase-2 step.

**Two conventions this file fixes, because the paper leaves them open.**

*Jacobian orientation in Eq. 9.* The equation writes `J_z J_z^T + eps I_k` with `I_k` the
k x k identity, where k is the latent dimension. The decoder maps R^k -> R^n, so its Jacobian
is (n, k) and `J J^T` would be (n, n), not (k, k). For the stated shape the Gram matrix must be
the k x k one, so this file computes `J^T J + eps I_k`. That is also the quantity that measures
"how much the decoder locally stretches or shrinks the space", which is the paper's own gloss.

*Latent radius.* The paper states a 0.5-radius hypersphere explicitly in Sections 4 and 5,
and upstream hardcodes 0.5 as a literal in both autoencoder classes. The probe passes
`latent_radius=0.5`.

*Epsilon* genuinely is not stated. `eps` is 1e-6, recorded as our choice rather than the
authors'.

**The loss weights are the authors', with a caveat.** Their README gives sweep *ranges*:
recon [1.5, 2.0], hinge [0.5, 1.0], latent [1.0, 1.5], geom [0.025]. This file defaults to the
low end of each range, and geom is unambiguous at 0.025. We do not sweep; a single point from
the authors' own grid is more faithful than a guess, and less faithful than their swept result.

**The infeasible labelling is ours and is the part the paper cannot supply**, because our
feasible set is a dynamics manifold intersected with box bounds rather than an
inequality-defined region with an interior. See `infeasible_generators` below.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from made.baselines.fab_baseline import FABBaseline

# Appendix A.2 (Safety Gym setting) governs. The upstream README's own Safety Gym example
# command instead passes recon 1.5, feas 1.0, latent 1.0, geom 0.025, hinge 0.5 (Appendix
# A.1 grid's low end); the two sources disagree, and A.2's values are used here.
LAMBDA_RECON: float = 1.0
LAMBDA_FEASIBILITY: float = 0.5
LAMBDA_HINGE: float = 0.5
LAMBDA_LATENT: float = 0.5
LAMBDA_GEOM: float = 0.1
JACOBIAN_EPS: float = 1e-6

# Appendix A.1 states three; A.2 is silent, so this comes from upstream and A.1.
K_CRITIC_STEPS: int = 3

# Upstream caps the geometric term's sample count. Not stated in the paper.
GEOM_MAX_SAMPLES: int = 32


class Discriminator(eqx.Module):
    """`D_xi : I -> [0, 1]`, a standard feedforward network (paper, section 3.1).

    Architecture is not stated in the paper or its README, so this mirrors the encoder's
    shape — same width and depth — which keeps the two sides of phase 2 comparable in
    capacity rather than importing an arbitrary choice.
    """

    mlp: eqx.nn.MLP

    def __init__(
        self,
        state_dim: int,
        hidden: tuple[int, ...] = (256, 256),
        *,
        key: jax.Array,
    ):
        width = hidden[0] if hidden else max(2 * state_dim, 1)
        self.mlp = eqx.nn.MLP(
            in_size=2 * state_dim,
            out_size=1,
            width_size=width,
            depth=len(hidden),
            activation=jax.nn.relu,
            key=key,
        )

    def logit(self, pair: jax.Array) -> jax.Array:
        return self.mlp(pair)[0]

    def __call__(self, pair: jax.Array) -> jax.Array:
        return jax.nn.sigmoid(self.logit(pair))


def _pair(x_prev: jax.Array, x_curr: jax.Array) -> jax.Array:
    return jnp.concatenate([x_prev, x_curr])


def discriminator_loss(
    discriminator: Discriminator,
    pairs: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    """Eq. 5, as a numerically stable binary cross-entropy over logits.

    `labels` is 1 for feasible and 0 for infeasible, matching the paper's `c`.
    """
    logits = jax.vmap(discriminator.logit)(pairs)
    # Upstream weights the positive class by num_neg / max(num_pos, 1), via BCEWithLogitsLoss'
    # pos_weight. Our generators are balanced by construction so this is
    # expected to be a no-op at 1.0; it is implemented rather than skipped so that it stays
    # correct if the balance ever changes, and the realised value is reported.
    n_pos = jnp.sum(labels)
    pos_weight = (labels.shape[0] - n_pos) / jnp.maximum(n_pos, 1.0)
    # -(c log sigma(l) + (1-c) log(1 - sigma(l))), written via softplus so neither log
    # underflows: log sigma(l) = -softplus(-l) and log(1-sigma(l)) = -softplus(l).
    return jnp.mean(
        pos_weight * labels * jax.nn.softplus(-logits)
        + (1.0 - labels) * jax.nn.softplus(logits)
    )


def hinge_loss(model: FABBaseline, pairs: jax.Array, labels: jax.Array) -> jax.Array:
    """Eq. 7 — feasible points inside the sphere of radius r, infeasible outside."""
    state_dim = model.state_dim
    norms = jax.vmap(
        lambda p: jnp.linalg.norm(model.encode(p[:state_dim], p[state_dim:]))
    )(pairs)
    r = model.latent_radius
    return jnp.mean(
        labels * jax.nn.relu(norms - r) + (1.0 - labels) * jax.nn.relu(r - norms)
    )


def latent_loss(
    model: FABBaseline,
    discriminator: Discriminator,
    z_samples: jax.Array,
) -> jax.Array:
    """Eq. 8 — points sampled from the ball must decode to something the critic calls feasible.

    `-log D(R(z))`, written as `softplus(-logit)` for the same stability reason as Eq. 5.
    """

    def _one(z: jax.Array) -> jax.Array:
        x_prev, x_curr = model.decode(z)
        return jax.nn.softplus(-discriminator.logit(_pair(x_prev, x_curr)))

    return jnp.mean(jax.vmap(_one)(z_samples))


def geometric_loss(
    model: FABBaseline,
    z_samples: jax.Array,
    eps: float = JACOBIAN_EPS,
) -> jax.Array:
    """Eq. 9 — variance of `log det(J^T J + eps I_k)` over the ball.

    See the module docstring on the orientation: the decoder's Jacobian is (n, k) and the
    paper's `I_k` fixes the Gram matrix as the k x k one, so `J^T J` is used.

    This is the term that decides whether `latent_dim` was chosen correctly. With k above the
    intrinsic dimension of the feasible set the Gram matrix is rank-deficient, `log det` is
    dominated by `k_deficient * log eps`, and the loss measures the regulariser rather than any
    geometry — which is why k is fixed at 8 for the dynamic bicycle.
    """

    # Upstream (undisclosed by the paper) caps the term at 32 latent samples and filters to
    # finite, positive determinants before taking the variance; the cap changes the term's
    # value on batches larger than 32, and the filter drops degenerate determinants that
    # would otherwise make the variance non-finite. Both adopted here.
    z_samples = z_samples[:GEOM_MAX_SAMPLES]

    def _logdet(z: jax.Array) -> jax.Array:
        jac_prev, jac_curr = jax.jacfwd(model.decode)(z)
        jac = jnp.concatenate([jac_prev, jac_curr], axis=0)  # (n, k)
        gram = jac.T @ jac + eps * jnp.eye(jac.shape[1])
        sign, logabsdet = jnp.linalg.slogdet(gram)
        # The Gram matrix is positive definite by construction, so sign is +1; the guard keeps
        # a numerically negative determinant from producing a silent nan.
        return jnp.where(sign > 0, logabsdet, jnp.log(eps) * jac.shape[1])

    logdets = jax.vmap(_logdet)(z_samples)
    # Upstream's finite-and-positive filter. Written as a masked variance rather than a boolean
    # index because the shape must stay static under jit.
    keep = jnp.isfinite(logdets)
    n_keep = jnp.sum(keep)
    mean = jnp.where(n_keep > 0, jnp.sum(jnp.where(keep, logdets, 0.0)) / jnp.maximum(n_keep, 1),
                     0.0)
    var = jnp.where(
        n_keep > 0,
        jnp.sum(jnp.where(keep, (logdets - mean) ** 2, 0.0)) / jnp.maximum(n_keep, 1),
        0.0,
    )
    return var


def feasibility_loss(
    model: FABBaseline,
    discriminator: Discriminator,
    feasible_pairs: jax.Array,
) -> jax.Array:
    """`L_feasibility` -- a fifth term with no equation in the paper.

    `lambda_feasibility` appears in Appendix A.2 and the upstream CLI, but Eq. 6 enumerates
    only four weights (`lambda_recon, lambda_latent, lambda_hinge, lambda_geom`) and
    Algorithm 1 line 23 names only four terms; this term exists in upstream's code and
    hyperparameters, not its mathematics.

    Upstream (`training.py`) targets the Safety Gym branch at ONES, not the input's own
    label (Appendix A.2 sets Safety Gym as the setting), pushing the reconstruction to be
    called feasible regardless of the input's own label -- the generic (non-Safety-Gym)
    branch instead targets that label. Followed here.

    Not a rename of `L_latent`: `L_latent` scores decodes of ball draws, `R(z)` for `z ~ S`;
    this scores the reconstruction path, `R(E(y, x))`, of a real input.
    """
    state_dim = model.state_dim
    def _one(pair: jax.Array) -> jax.Array:
        pred = model(pair[:state_dim], pair[state_dim:])
        return jax.nn.softplus(-discriminator.logit(_pair(*pred)))

    return jnp.mean(jax.vmap(_one)(feasible_pairs))


class StrucTerms(NamedTuple):
    """The four terms of Eq. 6, kept separate so each is logged rather than only the total."""

    recon: jax.Array
    feasibility: jax.Array
    hinge: jax.Array
    latent: jax.Array
    geom: jax.Array
    total: jax.Array


def structuring_loss(
    model: FABBaseline,
    discriminator: Discriminator,
    pairs: jax.Array,
    labels: jax.Array,
    feasible_pairs: jax.Array,
    z_samples: jax.Array,
    *,
    lambda_recon: float = LAMBDA_RECON,
    lambda_feasibility: float = LAMBDA_FEASIBILITY,
    lambda_hinge: float = LAMBDA_HINGE,
    lambda_latent: float = LAMBDA_LATENT,
    lambda_geom: float = LAMBDA_GEOM,
) -> StrucTerms:
    """Eq. 6. `L_recon` is computed on the FEASIBLE pairs only, as in phase 1.

    Algorithm 1 line 20 computes `L_recon` via Eq. 4, which is defined over `T_feas`; the
    reconstruction target for an infeasible point is not defined by the paper and asking the
    decoder to reproduce one would contradict `L_latent`.
    """
    state_dim = model.state_dim
    pred = jax.vmap(lambda p: model(p[:state_dim], p[state_dim:]))(feasible_pairs)
    recon = jnp.mean(
        (jnp.concatenate(pred, axis=-1) - feasible_pairs) ** 2
    )
    feasibility = feasibility_loss(model, discriminator, feasible_pairs)
    hinge = hinge_loss(model, pairs, labels)
    latent = latent_loss(model, discriminator, z_samples)
    geom = geometric_loss(model, z_samples)
    total = (
        lambda_recon * recon
        + lambda_feasibility * feasibility
        + lambda_hinge * hinge
        + lambda_latent * latent
        + lambda_geom * geom
    )
    return StrucTerms(recon=recon, feasibility=feasibility, hinge=hinge, latent=latent,
                      geom=geom, total=total)


def sample_ball(key: jax.Array, n: int, latent_dim: int, radius: float) -> jax.Array:
    """Uniform samples from `S := {z : ||z||_2 <= r}`, the set the paper projects onto."""
    dir_key, rad_key = jax.random.split(key)
    directions = jax.random.normal(dir_key, (n, latent_dim))
    directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)
    # r * u^(1/d) gives a uniform density in the ball rather than a shell.
    radii = radius * jax.random.uniform(rad_key, (n, 1)) ** (1.0 / latent_dim)
    return directions * radii


# ---------------------------------------------------------------------------
# The infeasible labelling. Ours, not the paper's.
# ---------------------------------------------------------------------------

InfeasibleGenerator = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


def make_infeasible_generators(
    state_min: jax.Array,
    state_max: jax.Array,
    perturbation_scale_per_dim: jax.Array,
    step_fn: Callable[[jax.Array, jax.Array], jax.Array],
    control_min: jax.Array,
    control_max: jax.Array,
) -> dict[str, InfeasibleGenerator]:
    """Four generators in equal share, each targeting a different way to be infeasible.

    Generators 1 and 2 use the EVALUATION PROTOCOL's own per-dimension scales, so
    the discriminator is trained on the same perturbation distribution the evaluation applies.
    **The corrected model is never given that advantage.** That asymmetry is deliberate and
    must be disclosed wherever the comparison appears.

    `step_fn(x_prev, u) -> x_curr` is the one-step forward map; it defines the manifold that
    generators 2 and 3 move off.
    """
    def box_violating(key: jax.Array, x_prev: jax.Array, x_curr: jax.Array) -> jax.Array:
        """Push `x_curr` just past a bound, at the evaluation protocol's scales.

        Dimensions whose bound is infinite cannot be violated and are left at `x_curr`.
        """
        sign_key, mag_key = jax.random.split(key)
        direction = jnp.sign(jax.random.normal(sign_key, x_curr.shape))
        excess = perturbation_scale_per_dim * jnp.abs(
            jax.random.normal(mag_key, x_curr.shape)
        )
        bound = jnp.where(direction > 0, state_max, state_min)
        beyond = bound + direction * excess
        return jnp.where(jnp.isfinite(bound), beyond, x_curr)

    def off_manifold(key: jax.Array, x_prev: jax.Array, x_curr: jax.Array) -> jax.Array:
        """Move `x_curr` off the reachable set while keeping it inside the box."""
        noise = perturbation_scale_per_dim * jax.random.normal(key, x_curr.shape)
        return jnp.clip(x_curr + noise, state_min, state_max)

    def wrong_control(key: jax.Array, x_prev: jax.Array, x_curr: jax.Array) -> jax.Array:
        """Integrate a DIFFERENT admissible control from the same anchor.

        The load-bearing generator: the result satisfies every bound and is physically
        plausible in isolation, differing from a feasible pair only in the control that
        produced it. A discriminator that separates these has learned the manifold; one that
        separates only the box violations has learned the box.
        """
        u = jax.random.uniform(
            key, control_min.shape, minval=control_min, maxval=control_max
        )
        return step_fn(x_prev, u)

    def ambient(key: jax.Array, x_prev: jax.Array, x_curr: jax.Array) -> jax.Array:
        """Draw `x_curr` uniformly in the box. Infeasible with probability 1."""
        lo = jnp.where(jnp.isfinite(state_min), state_min, -jnp.ones_like(state_min))
        hi = jnp.where(jnp.isfinite(state_max), state_max, jnp.ones_like(state_max))
        return jax.random.uniform(key, x_curr.shape, minval=lo, maxval=hi)

    return {
        "box_violating": box_violating,
        "off_manifold": off_manifold,
        "wrong_control": wrong_control,
        "ambient": ambient,
    }
