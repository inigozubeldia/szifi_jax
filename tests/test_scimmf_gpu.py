"""GPU one-dep sciMMF q-maps must match sequential get_mmf_q_map.

The batched kernel is the CNC-critical piece: same inv_cov, templates, and
(a·C^{-1}b)/(b·C^{-1}b) constraint as the per-tile path.
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from szifi_jax import maps, mmf


def _spd_inv(n_tile, nx, n_freq, rng):
    a = rng.standard_normal((n_tile, nx, nx, n_freq, n_freq))
    cov = np.einsum("txyij,txykj->txyik", a, a) + 3.0 * np.eye(n_freq)
    return jnp.asarray(np.linalg.inv(cov))


def test_gpu_scimmf_q_maps_match_sequential():
    rng = np.random.default_rng(0)
    n_tile, nx, n_freq, n_theta = 2, 32, 6, 3
    pix = maps.pixel(nx, 0.5 / 60.0 / 180.0 * np.pi)
    tmaps = jnp.asarray(rng.standard_normal((n_tile, nx, nx, n_freq)))
    tem_list = [jnp.asarray(rng.standard_normal((nx, nx, n_freq))) for _ in range(n_theta)]
    inv = _spd_inv(n_tile, nx, n_freq, rng)
    a_vec = jnp.asarray(np.array([-4.0, -2.5, 0.0, 3.0, 8.0, 12.0]))
    b_vec = jnp.asarray(np.array([0.2, 0.5, 1.0, 3.0, 20.0, 80.0]))
    a_dot_b = jnp.asarray(rng.standard_normal((n_tile, nx, nx)))
    b_dot_b = jnp.asarray(1.5 + rng.random((n_tile, nx, nx)))
    a_matrix = jnp.stack([a_vec, b_vec], axis=1)

    q_gpu, y_gpu, std_gpu = mmf.get_mmf_q_maps_tiles(
        tmaps, tem_list, inv, pix, a_vec, b_vec=b_vec, a_dot_b=a_dot_b, b_dot_b=b_dot_b
    )

    q_seq = []
    y_seq = []
    std_seq = []
    for t in range(n_tile):
        prec = types.SimpleNamespace(
            cmmf_type="one_dep",
            a_matrix=a_matrix,
            a_dot_b=a_dot_b[t],
            b_dot_b=b_dot_b[t],
        )
        qs, ys, ss = [], [], []
        for tem in tem_list:
            q, y, s = mmf.get_mmf_q_map(
                tmaps[t],
                tem,
                inv[t],
                pix,
                mmf_type="spectrally_constrained",
                cmmf_prec=prec,
                comp=0,
            )
            qs.append(q)
            ys.append(y)
            ss.append(s)
        q_seq.append(jnp.stack(qs))
        y_seq.append(jnp.stack(ys))
        std_seq.append(jnp.stack(ss))
    q_seq = jnp.stack(q_seq)
    y_seq = jnp.stack(y_seq)
    std_seq = jnp.stack(std_seq)

    q_rel = float(jnp.max(jnp.abs(q_gpu - q_seq)) / jnp.max(jnp.abs(q_seq)))
    y_rel = float(jnp.max(jnp.abs(y_gpu - y_seq)) / jnp.max(jnp.abs(y_seq)))
    s_rel = float(jnp.max(jnp.abs(std_gpu - std_seq)) / jnp.max(jnp.abs(std_seq)))
    assert q_rel < 1e-10, f"q rel {q_rel}"
    assert y_rel < 1e-10, f"y rel {y_rel}"
    assert s_rel < 1e-10, f"std rel {s_rel}"


def test_gpu_mmf_without_b_vec_still_standard():
    rng = np.random.default_rng(1)
    n_tile, nx, n_freq = 2, 16, 6
    pix = maps.pixel(nx, 0.5 / 60.0 / 180.0 * np.pi)
    tmaps = jnp.asarray(rng.standard_normal((n_tile, nx, nx, n_freq)))
    tem_list = [jnp.asarray(rng.standard_normal((nx, nx, n_freq)))]
    inv = _spd_inv(n_tile, nx, n_freq, rng)
    a_vec = jnp.ones(n_freq)
    q_std, _, _ = mmf.get_mmf_q_maps_tiles(tmaps, tem_list, inv, pix, a_vec)
    q_sci, _, _ = mmf.get_mmf_q_maps_tiles(
        tmaps, tem_list, inv, pix, a_vec, b_vec=a_vec,
        a_dot_b=jnp.zeros((n_tile, nx, nx)),
        b_dot_b=jnp.ones((n_tile, nx, nx)),
    )
    # Constraint term is zero when a_dot_b=0, so sciMMF must reduce to standard.
    rel = float(jnp.max(jnp.abs(q_sci - q_std)) / jnp.max(jnp.abs(q_std)))
    assert rel < 1e-10, f"zero-constraint sciMMF != standard, rel {rel}"


def test_gpu_scimmf_q_maps_match_sequential_float32():
    rng = np.random.default_rng(0)
    n_tile, nx, n_freq, n_theta = 2, 32, 6, 3
    pix = maps.pixel(nx, 0.5 / 60.0 / 180.0 * np.pi)
    tmaps = jnp.asarray(rng.standard_normal((n_tile, nx, nx, n_freq)), dtype=jnp.float32)
    tem_list = [jnp.asarray(rng.standard_normal((nx, nx, n_freq)), dtype=jnp.float32) for _ in range(n_theta)]
    inv = _spd_inv(n_tile, nx, n_freq, rng).astype(jnp.float64)
    a_vec = jnp.asarray(np.array([-4.0, -2.5, 0.0, 3.0, 8.0, 12.0]))
    b_vec = jnp.asarray(np.array([0.2, 0.5, 1.0, 3.0, 20.0, 80.0]))
    a_dot_b = jnp.asarray(rng.standard_normal((n_tile, nx, nx)))
    b_dot_b = jnp.asarray(1.5 + rng.random((n_tile, nx, nx)))
    a_matrix = jnp.stack([a_vec, b_vec], axis=1)

    q_gpu, _, _ = mmf.get_mmf_q_maps_tiles(
        tmaps, tem_list, inv, pix, a_vec, b_vec=b_vec, a_dot_b=a_dot_b, b_dot_b=b_dot_b
    )
    q_seq = []
    for t in range(n_tile):
        prec = types.SimpleNamespace(
            cmmf_type="one_dep",
            a_matrix=a_matrix,
            a_dot_b=a_dot_b[t],
            b_dot_b=b_dot_b[t],
        )
        qs = []
        for tem in tem_list:
            q, _, _ = mmf.get_mmf_q_map(
                tmaps[t], tem, inv[t], pix,
                mmf_type="spectrally_constrained", cmmf_prec=prec, comp=0,
            )
            qs.append(q)
        q_seq.append(jnp.stack(qs))
    q_seq = jnp.stack(q_seq)
    q_rel = float(jnp.max(jnp.abs(q_gpu - q_seq)) / jnp.max(jnp.abs(q_seq)))
    assert q_rel < 1e-6, f"float32 q rel {q_rel}"


def test_one_dep_dots_match_scmmf_precomputation():
    from szifi_jax import expt

    rng = np.random.default_rng(2)
    nx, n_freq = 32, 6
    pix = maps.pixel(nx, 0.5 / 60.0 / 180.0 * np.pi)
    exp = expt.experiment("Planck_simple")
    a_matrix = np.stack(
        [np.asarray(exp.tsz_sed), np.array([0.2, 0.5, 1.0, 3.0, 20.0, 80.0])],
        axis=1,
    )
    inv = _spd_inv(1, nx, n_freq, rng)[0]
    freqs = np.arange(n_freq)
    lrange = [0, 1000000]
    prec = mmf.scmmf_precomputation(
        pix=pix,
        freqs=freqs,
        inv_cov=inv,
        lrange=lrange,
        beam_type="gaussian",
        exp=exp,
        cmmf_type="one_dep",
        a_matrix=a_matrix,
        mmf_type="spectrally_constrained",
    )
    a_fft, b_fft = mmf.scmmf_one_dep_sed_ffts(
        pix=pix,
        freqs=freqs,
        inv_cov=inv,
        lrange=lrange,
        beam_type="gaussian",
        exp=exp,
        a_matrix=a_matrix,
    )
    a_dot, b_dot = mmf.scmmf_one_dep_dots_tiles(a_fft, b_fft, inv[None, ...])
    a_rel = float(
        jnp.max(jnp.abs(a_dot[0] - prec.a_dot_b))
        / jnp.maximum(jnp.max(jnp.abs(prec.a_dot_b)), 1e-30)
    )
    b_rel = float(
        jnp.max(jnp.abs(b_dot[0] - prec.b_dot_b))
        / jnp.maximum(jnp.max(jnp.abs(prec.b_dot_b)), 1e-30)
    )
    assert a_rel < 1e-10, f"a_dot_b rel {a_rel}"
    assert b_rel < 1e-10, f"b_dot_b rel {b_rel}"
