

# The unbounded output map, behind a flag that defaults off.


def _encoder(unbounded: bool, *, lref_residual: bool = False, param_dim: int = 1,
             seed: int = 0):
    import jax
    import jax.numpy as jnp

    from made.models.encoder import MetadataEncoder

    return MetadataEncoder(
        metadata_dim=5,
        param_dim=param_dim,
        hidden=(16, 16),
        param_scales=jnp.ones((param_dim,)),
        num_locations=4,
        embedding_dim=8,
        location_id_index=4,
        unbounded_scale=unbounded,
        lref_residual=lref_residual,
        key=jax.random.key(seed),
    )


def test_encoder_unbounded_flag_off_is_bit_identical_to_the_published_map():
    """Flag OFF must reproduce `sigmoid(mlp(...)) * param_scales` exactly.

    Every existing checkpoint -- E01 (encoder off everywhere) and the canonical inD models --
    was trained under that map, so a change here would silently rescore them.
    """
    import jax
    import jax.numpy as jnp

    enc = _encoder(False)
    meta = jnp.asarray([5.0, 2.0, 1.0, 0.0, 3.0])
    got = enc(meta)
    # The published map, recomputed here rather than imported, so the test would fail if the
    # module's own definition drifted.
    idx = enc.location_id_index
    float_cols = jnp.concatenate([meta[:idx], meta[idx + 1 :]], axis=0)
    emb = enc.embedding(jnp.asarray(meta[idx], dtype=jnp.int32) - 1)
    expected = jax.nn.sigmoid(enc.mlp(jnp.concatenate([float_cols, emb], axis=0))) * enc.param_scales
    assert jnp.array_equal(got, expected)
    assert float(got[0]) < 1.0  # measured bound on the published map


def test_encoder_unbounded_flag_on_can_exceed_one():
    """Flag ON must be able to emit a wheelbase above 1.0 m, which the sigmoid map never can.

    Asserted on the MAP rather than on a trained model: a random init need not exceed 1.0, but
    softplus must be capable of it, and the sigmoid map must not be.
    """
    import jax
    import jax.numpy as jnp

    on, off = _encoder(True), _encoder(False)
    # A metadata vector is not enough to force a large pre-activation, so drive the map directly.
    raw = jnp.asarray([4.0])
    assert float(jax.nn.softplus(raw)[0]) > 1.0
    assert float(jax.nn.sigmoid(raw)[0] * off.param_scales[0]) < 1.0

    # And the module honours the flag end to end, on the same metadata and key.
    meta = jnp.asarray([5.0, 2.0, 1.0, 0.0, 3.0])
    assert not jnp.array_equal(on(meta), off(meta))
    assert float(on(meta)[0]) > 0.0  # a wheelbase stays positive under softplus


# L = L_REF + signed residual, replacing the earlier softplus head.


def test_lref_residual_equals_l_ref_when_the_mlp_outputs_zero():
    """With a zero MLP output the encoder must return exactly L_REF, the scorers' own constant.

    This is what makes the design a RESIDUAL: day one is the known model, not a guess near it.
    The constant is imported, never written a second time, so this fails if the two ever drift.
    """
    import equinox as eqx
    import jax.numpy as jnp

    from made.evaluation.real_data_eval import L_REF

    enc = _encoder(False, lref_residual=True)
    # Zero the final layer so the MLP emits exactly 0 for any input.
    last = enc.mlp.layers[-1]
    enc = eqx.tree_at(
        lambda e: (e.mlp.layers[-1].weight, e.mlp.layers[-1].bias),
        enc,
        (jnp.zeros_like(last.weight), jnp.zeros_like(last.bias)),
    )
    meta = jnp.asarray([5.0, 2.0, 1.0, 0.0, 3.0])
    assert float(enc(meta)[0]) == L_REF


def test_lref_residual_can_go_below_and_above_l_ref():
    """The residual is SIGNED and unbounded: a negative MLP output must lower L below L_REF.

    Asserted by driving the final layer's bias, because a random init need not straddle L_REF.
    This design adds no clamp, so nothing should stop L from going anywhere -- including, in
    principle, below zero, which the report is asked to surface rather than the code to prevent.
    """
    import equinox as eqx
    import jax.numpy as jnp

    from made.evaluation.real_data_eval import L_REF

    base = _encoder(False, lref_residual=True)
    meta = jnp.asarray([5.0, 2.0, 1.0, 0.0, 3.0])

    def with_bias(value: float):
        last = base.mlp.layers[-1]
        return eqx.tree_at(
            lambda e: (e.mlp.layers[-1].weight, e.mlp.layers[-1].bias),
            base,
            (jnp.zeros_like(last.weight), jnp.full_like(last.bias, value)),
        )

    assert float(with_bias(+1.5)(meta)[0]) > L_REF
    assert float(with_bias(-1.5)(meta)[0]) < L_REF
    # No clamp: a large enough negative residual drives L negative rather than being floored.
    assert float(with_bias(-(L_REF + 1.0))(meta)[0]) < 0.0
