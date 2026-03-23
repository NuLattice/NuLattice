import pytest
import numpy as np
import torch

import NuLattice.CCM.coupled_cluster as legacy_cc
import NuLattice.CCM.ccDgrams as legacy_dgrams
import NuLattice.CCM.soa.ccDgrams as soa_dgrams


def to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


def to_torch(arr, device="cpu", dtype=torch.float64):
    return torch.tensor(arr, device=device, dtype=dtype)


def assert_dgram_match(legacy_val, new_val, name, tol=1e-10):
    new_np = to_numpy(new_val)
    if not np.allclose(legacy_val, new_np, atol=tol, rtol=tol):
        diff = np.abs(legacy_val - new_np)
        pytest.fail(
            f"Diagram {name} mismatch!\n"
            f"Max Diff: {np.max(diff):.2e}\n"
            f"Legacy Sample: {legacy_val.flatten()[:3]}\n"
            f"New Sample:    {new_np.flatten()[:3]}"
        )


@pytest.fixture
def test_tensors():
    """Generates consistent random tensors for testing contractions."""
    p, h = 4, 4
    np.random.seed(42)

    return {
        "t1": np.random.randn(p, h),
        "t2": np.random.randn(p, p, h, h),
        "f_ph": np.random.randn(p, h),
        "f_pp": np.random.randn(p, p),
        "f_hh": np.random.randn(h, h),
        "v_phph": np.random.randn(p, h, p, h),
        "v_phhh": np.random.randn(p, h, h, h),
        "v_pphh": np.random.randn(p, p, h, h),
        "v_hhhh": np.random.randn(h, h, h, h),
    }


@pytest.mark.parametrize(
    "dgram_name",
    [
        "dgram_akci_ck",  # H1 += -V*t1
        "dgram_ck_acik",  # H1 += f*t2
        "dgram_cikl_cakl",  # H1 += -0.5*V*t2
    ],
)
def test_t1_h1_diagrams(test_tensors, dgram_name):
    d = test_tensors
    t1_t, t2_t = to_torch(d["t1"]), to_torch(d["t2"])

    dgram_to_v = {
        "dgram_akci_ck": ("v_phph", d["v_phph"]),
        "dgram_ck_acik": ("f_ph", d["f_ph"]),
        "dgram_cikl_cakl": ("v_phhh", d["v_phhh"]),
    }

    v_label, v_leg = dgram_to_v[dgram_name]
    v_soa = to_torch(v_leg)

    leg_func = getattr(legacy_dgrams, dgram_name)
    soa_func = getattr(soa_dgrams, dgram_name)

    if dgram_name == "dgram_akci_ck":
        res_leg = leg_func(v_leg, d["t1"])
        res_soa = soa_func(v_soa, t1_t)
    else:
        res_leg = leg_func(v_leg, d["t2"])
        res_soa = soa_func(v_soa, t2_t)

    assert_dgram_match(res_leg, res_soa, dgram_name)


def test_t1_three_body_dgram(test_tensors):
    """Specific test for the N^6 diagram dgram_cdkl_ck_dali."""
    d = test_tensors
    res_leg = legacy_dgrams.dgram_cdkl_ck_dali(d["v_pphh"], d["t1"], d["t2"])
    res_soa = soa_dgrams.dgram_cdkl_ck_dali(
        to_torch(d["v_pphh"]), to_torch(d["t1"]), to_torch(d["t2"])
    )
    assert_dgram_match(res_leg, res_soa, "dgram_cdkl_ck_dali")


@pytest.mark.parametrize(
    "dgram_name",
    [
        "dgram_ck_ci",  # X_hh += -0.5*f*t1
        "dgram_ck_ak",  # X_pp += -0.5*f*t1
        "dgram_bijk_bj",  # X_hh += -V*t1
        "dgram_cdlk_cdli",  # X_hh += -0.5*V*t2
        "dgram_dckl_dakl",  # X_pp += -0.5*V*t2
    ],
)
def test_intermediate_dressing(test_tensors, dgram_name):
    d = test_tensors
    t1_t, t2_t = to_torch(d["t1"]), to_torch(d["t2"])

    input_map = {
        "dgram_ck_ci": (d["f_ph"], t1_t),
        "dgram_ck_ak": (d["f_ph"], t1_t),
        "dgram_bijk_bj": (d["v_phhh"], t1_t),
        "dgram_cdlk_cdli": (d["v_pphh"], t2_t),
        "dgram_dckl_dakl": (d["v_pphh"], t2_t),
    }

    v_leg, arg2_soa = input_map[dgram_name]
    v_soa = to_torch(v_leg)
    arg2_leg = to_numpy(arg2_soa)

    res_leg = getattr(legacy_dgrams, dgram_name)(v_leg, arg2_leg)
    res_soa = getattr(soa_dgrams, dgram_name)(v_soa, arg2_soa)

    assert_dgram_match(res_leg, res_soa, dgram_name)


def test_t1_squared_intermediates(test_tensors):
    """Verify X_hh and X_pp terms containing T1*T1."""
    d = test_tensors
    v_leg, t1_leg = d["v_pphh"], d["t1"]
    v_soa, t1_soa = to_torch(v_leg), to_torch(t1_leg)

    res_leg_hh = legacy_dgrams.dgram_cdlk_cl_di(v_leg, t1_leg)
    res_soa_hh = soa_dgrams.dgram_cdlk_cl_di(v_soa, t1_soa)
    assert_dgram_match(res_leg_hh, res_soa_hh, "dgram_cdlk_cl_di")

    res_leg_pp = legacy_dgrams.dgram_cdkl_dk_al(v_leg, t1_leg)
    res_soa_pp = soa_dgrams.dgram_cdkl_dk_al(v_soa, t1_soa)
    assert_dgram_match(res_leg_pp, res_soa_pp, "dgram_cdkl_dk_al")


def test_t2_v_hhhh_dgram(test_tensors):
    d = test_tensors
    res_leg = legacy_dgrams.dgram_klij_abkl(d["v_hhhh"], d["t2"])
    res_soa = soa_dgrams.dgram_klij_abkl(to_torch(d["v_hhhh"]), to_torch(d["t2"]))
    assert_dgram_match(res_leg, res_soa, "dgram_klij_abkl")


def test_t2_v_phph_dgram(test_tensors):
    d = test_tensors
    res_leg = legacy_dgrams.dgram_bkcj_acik(d["v_phph"], d["t2"])
    res_soa = soa_dgrams.dgram_bkcj_acik(to_torch(d["v_phph"]), to_torch(d["t2"]))
    assert_dgram_match(res_leg, res_soa, "dgram_bkcj_acik")


def test_permutators():
    p, h = 2, 2
    val = torch.randn(p, p, h, h)
    val_np = to_numpy(val)

    assert_dgram_match(legacy_dgrams.pAB(val_np), soa_dgrams.pAB(val), "pAB Permutator")
    assert_dgram_match(legacy_dgrams.pIJ(val_np), soa_dgrams.pIJ(val), "pIJ Permutator")




@pytest.fixture
def physics_data():
    """Generates a small randomized system for bit-for-bit comparison."""
    p, h = 4, 4
    np.random.seed(42)
    return {
        "t1": np.random.randn(p, h) * 0.1,
        "t2": np.random.randn(p, p, h, h) * 0.1,
        "f_ph": np.random.randn(p, h),
        "f_pp": np.diag(np.random.rand(p) + 2.0),
        "f_hh": np.diag(np.random.rand(h) - 2.0),
        "v_phph": np.random.randn(p, h, p, h) * 0.1,
        "v_phhh": np.random.randn(p, h, h, h) * 0.1,
        "v_pphh": np.random.randn(p, p, h, h) * 0.1,
        "v_ppph": np.random.randn(p, p, p, h) * 0.1,
    }


def test_exact_t1iter_reproduction(physics_data):
    """
    Step-by-step reproduction of the legacy t1Iter logic.
    Identifies if discrepancies appear in Diagrams, Intermediates, or Denominators.
    """
    d = physics_data
    pnum, hnum = d["t1"].shape
    t1_t, t2_t = to_torch(d["t1"]), to_torch(d["t2"])
    f_ph_t, f_pp_t, f_hh_t = (
        to_torch(d["f_ph"]),
        to_torch(d["f_pp"]),
        to_torch(d["f_hh"]),
    )
    v_phph_t, v_phhh_t, v_pphh_t, v_ppph_t = (
        to_torch(d["v_phph"]),
        to_torch(d["v_phhh"]),
        to_torch(d["v_pphh"]),
        to_torch(d["v_ppph"]),
    )

    t1_leg_input = d["t1"].copy()
    t1_leg_final = legacy_cc.t1Iter(
        t1_leg_input,
        d["t2"],
        d["f_ph"],
        d["f_pp"],
        d["f_hh"],
        d["v_phph"],
        d["v_phhh"],
        d["v_pphh"],
        d["v_ppph"],
        sparse=False,
    )

    H1 = f_ph_t.clone()
    H1 += soa_dgrams.dgram_akci_ck(v_phph_t, t1_t)
    H1 += soa_dgrams.dgram_ck_acik(f_ph_t, t2_t)
    H1 += soa_dgrams.dgram_cikl_cakl(v_phhh_t, t2_t)
    H1 += soa_dgrams.dgram_cdkl_ck_dali(v_pphh_t, t1_t, t2_t)

    leg_H1_part1 = (
        d["f_ph"]
        + legacy_dgrams.dgram_akci_ck(d["v_phph"], d["t1"])
        + legacy_dgrams.dgram_ck_acik(d["f_ph"], d["t2"])
        + legacy_dgrams.dgram_cikl_cakl(d["v_phhh"], d["t2"])
        + legacy_dgrams.dgram_cdkl_ck_dali(d["v_pphh"], d["t1"], d["t2"])
    )
    assert np.allclose(leg_H1_part1, to_numpy(H1), atol=1e-12), (
        "Mismatch in H1 Base Diagrams"
    )

    X_hh = -f_hh_t.clone()
    X_pp = f_pp_t.clone()
    X_hh += soa_dgrams.dgram_ck_ci(f_ph_t, t1_t)
    X_pp += soa_dgrams.dgram_ck_ak(f_ph_t, t1_t)
    X_hh += soa_dgrams.dgram_bijk_bj(v_phhh_t, t1_t)
    X_hh += soa_dgrams.dgram_cdlk_cdli(v_pphh_t, t2_t)
    X_pp += soa_dgrams.dgram_dckl_dakl(v_pphh_t, t2_t)
    X_hh += soa_dgrams.dgram_cdlk_cl_di(v_pphh_t, t1_t)
    X_pp += soa_dgrams.dgram_cdkl_dk_al(v_pphh_t, t1_t)

    H1 += -0.5 * torch.einsum("cdak, cdki -> ai", v_ppph_t, t2_t)
    X_pp -= torch.einsum("cdak, ck -> ad", v_ppph_t, t1_t)

    leg_X_hh = (
        -d["f_hh"]
        + legacy_dgrams.dgram_ck_ci(d["f_ph"], d["t1"])
        + legacy_dgrams.dgram_bijk_bj(d["v_phhh"], d["t1"])
        + legacy_dgrams.dgram_cdlk_cdli(d["v_pphh"], d["t2"])
        + legacy_dgrams.dgram_cdlk_cl_di(d["v_pphh"], d["t1"])
    )

    assert np.allclose(leg_X_hh, to_numpy(X_hh), atol=1e-12), (
        "Mismatch in X_hh intermediate"
    )

    H1 += torch.einsum("ac, ci -> ai", X_pp, t1_t)
    H1 += torch.einsum("ki, ak -> ai", X_hh, t1_t)

    diag_h = torch.diagonal(X_hh)
    diag_p = torch.diagonal(X_pp)
    denom = -(diag_p.unsqueeze(1) + diag_h.unsqueeze(0))

    t1_torch_final = t1_t + (H1 / denom)

    diff = np.abs(to_numpy(t1_torch_final) - t1_leg_final)
    print(f"\nMax Exact Reproduction Diff: {diff.max():.2e}")

    assert np.allclose(t1_leg_final, to_numpy(t1_torch_final), atol=1e-12), (
        f"Step-by-step reproduction failed with max diff {diff.max():.2e}"
    )
