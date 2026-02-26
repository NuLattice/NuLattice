import pytest
import numpy as np
import torch
from opt_einsum import contract

import NuLattice.CCM.coupled_cluster as legacy_cc
import NuLattice.CCM.ccDgrams as legacy_dgrams
import NuLattice.CCM.soa.coupled_cluster as soa_cc
import NuLattice.CCM.soa.ccDgrams as soa_dgrams

def to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t

def to_torch(arr, device=None, dtype=torch.float64):
    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=dtype)
    return torch.tensor(arr, device=device, dtype=dtype)

def assert_tensors_match(a, b, name, tol=1e-7):
    arr_new = to_numpy(b)
    assert a.shape == arr_new.shape, f"Shape mismatch in {name}: {a.shape} vs {arr_new.shape}"
    if not np.allclose(a, arr_new, atol=tol, rtol=tol):
        diff = np.abs(a - arr_new)
        max_diff = np.max(diff)
        mismatch_idx = np.where(diff > tol)
        print(f"\n[FAIL] {name} max diff: {max_diff:.2e}")
        print(f"Indices of mismatch (first 3): {list(zip(*mismatch_idx))[:3]}")
        print(f"A vals: {to_numpy(a[mismatch_idx][:3])}")
        print(f"B vals: {to_numpy(arr_new[mismatch_idx][:3])}")
        pytest.fail(f"{name} mismatch")

@pytest.fixture
def dummy_data():
    """Generates random but consistent physics inputs for P=4, H=4 system."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    p, h = 4, 4
    
    t1 = np.random.randn(p, h) * 0.1
    t2 = np.random.randn(p, p, h, h) * 0.05
    
    f_pp = np.diag(np.random.rand(p) + 2.0) # Ensure p energies > h
    f_hh = np.diag(np.random.rand(h) - 2.0)
    f_ph = np.random.randn(p, h) * 0.1
    
    v_phph = np.random.randn(p, h, p, h) * 0.1
    v_phhh = np.random.randn(p, h, h, h) * 0.1
    v_pphh = np.random.randn(p, p, h, h) * 0.1
    
    # v_ppph is the critical one for Sparse/Dense checks
    v_ppph = np.random.randn(p, p, p, h) * 0.1
    
    
    return {
        "t1": t1, "t2": t2,
        "f_pp": f_pp, "f_ph": f_ph, "f_hh": f_hh,
        "v_phph": v_phph, "v_phhh": v_phhh, "v_pphh": v_pphh,
        "v_ppph": v_ppph
    }


def test_01_soa_sparse_dense_consistency(dummy_data):
    """
    INTERNAL CONSISTENCY: Does soa_cc.t1Iter give the same result 
    regardless of sparse=True or sparse=False? 
    This verifies the if/else block in the new code.
    """
    d = dummy_data
    
    t1 = to_torch(d['t1'])
    t2 = to_torch(d['t2'])
    f_ph = to_torch(d['f_ph'])
    f_pp = to_torch(d['f_pp'])
    f_hh = to_torch(d['f_hh'])
    v_phph = to_torch(d['v_phph'])
    v_phhh = to_torch(d['v_phhh'])
    v_pphh = to_torch(d['v_pphh'])
    v_ppph_dense = to_torch(d['v_ppph'])
    
    res_dense = soa_cc.t1Iter(
        t1, t2, f_ph, f_pp, f_hh, v_phph, v_phhh, v_pphh, v_ppph_dense, sparse=False
    )
    
    indices = torch.nonzero(v_ppph_dense).T
    values = v_ppph_dense[indices[0], indices[1], indices[2], indices[3]]
    v_ppph_soa = (indices, values)
    
    v_ppph_results_sparse = soa_dgrams.v_ppph_dgrams(v_ppph_soa, t1, t2)
    
    res_sparse = soa_cc.t1Iter(
        t1, t2, f_ph, f_pp, f_hh, v_phph, v_phhh, v_pphh, v_ppph_results_sparse, sparse=True
    )
    
    assert_tensors_match(to_numpy(res_dense), to_numpy(res_sparse), "SoA Internal Sparse-Dense Consistency")


def test_02_legacy_vs_soa_component_h1(dummy_data):
    """
    COMPONENT CHECK: Verify the H1 (Numerator) construction matches Legacy.
    This specifically checks the sum of diagrams excluding intermediates X.
    """
    d = dummy_data
    
    H1_legacy = d['f_ph'].copy()
    H1_legacy += legacy_dgrams.dgram_akci_ck(d['v_phph'], d['t1'])
    H1_legacy += legacy_dgrams.dgram_ck_acik(d['f_ph'], d['t2'])
    H1_legacy += legacy_dgrams.dgram_cikl_cakl(d['v_phhh'], d['t2'])
    H1_legacy += legacy_dgrams.dgram_cdkl_ck_dali(d['v_pphh'], d['t1'], d['t2'])
    
    t1 = to_torch(d['t1'])
    t2 = to_torch(d['t2'])
    f_ph = to_torch(d['f_ph'])
    v_phph = to_torch(d['v_phph'])
    v_phhh = to_torch(d['v_phhh'])
    v_pphh = to_torch(d['v_pphh'])

    H1_soa = f_ph.clone()
    H1_soa += soa_dgrams.dgram_akci_ck(v_phph, t1)
    H1_soa += soa_dgrams.dgram_ck_acik(f_ph, t2)
    H1_soa += soa_dgrams.dgram_cikl_cakl(v_phhh, t2)
    H1_soa += soa_dgrams.dgram_cdkl_ck_dali(v_pphh, t1, t2)
    
    assert_tensors_match(H1_legacy, H1_soa, "H1 Base Diagrams (Pre-Intermediates)")


def test_03_legacy_vs_soa_component_X(dummy_data):
    """
    COMPONENT CHECK: Verify X_pp and X_hh (Intermediates) construction matches Legacy.
    This checks the 'dressing' of the Fock matrix.
    """
    d = dummy_data
    
    pnum = len(d['f_pp'])
    hnum = len(d['f_hh'])
    
    X_hh_leg = np.zeros((hnum, hnum))
    X_pp_leg = np.zeros((pnum, pnum))
    
    X_hh_leg -= d['f_hh']
    X_pp_leg += d['f_pp']
    
    X_hh_leg += legacy_dgrams.dgram_ck_ci(d['f_ph'], d['t1'])
    X_pp_leg += legacy_dgrams.dgram_ck_ak(d['f_ph'], d['t1'])
    X_hh_leg += legacy_dgrams.dgram_bijk_bj(d['v_phhh'], d['t1'])
    X_hh_leg += legacy_dgrams.dgram_cdlk_cdli(d['v_pphh'], d['t2'])
    X_pp_leg += legacy_dgrams.dgram_dckl_dakl(d['v_pphh'], d['t2'])
    X_hh_leg += legacy_dgrams.dgram_cdlk_cl_di(d['v_pphh'], d['t1'])
    X_pp_leg += legacy_dgrams.dgram_cdkl_dk_al(d['v_pphh'], d['t1'])
    
    t1 = to_torch(d['t1'])
    t2 = to_torch(d['t2'])
    f_ph = to_torch(d['f_ph'])
    f_pp = to_torch(d['f_pp'])
    f_hh = to_torch(d['f_hh'])
    v_phhh = to_torch(d['v_phhh'])
    v_pphh = to_torch(d['v_pphh'])

    X_hh_soa = -f_hh.clone()
    X_pp_soa = f_pp.clone()
    
    X_hh_soa += soa_dgrams.dgram_ck_ci(f_ph, t1)
    X_pp_soa += soa_dgrams.dgram_ck_ak(f_ph, t1)
    X_hh_soa += soa_dgrams.dgram_bijk_bj(v_phhh, t1)
    X_hh_soa += soa_dgrams.dgram_cdlk_cdli(v_pphh, t2)
    X_pp_soa += soa_dgrams.dgram_dckl_dakl(v_pphh, t2)
    X_hh_soa += soa_dgrams.dgram_cdlk_cl_di(v_pphh, t1)
    X_pp_soa += soa_dgrams.dgram_cdkl_dk_al(v_pphh, t1)
    
    assert_tensors_match(X_hh_leg, X_hh_soa, "X_hh Intermediate Construction")
    assert_tensors_match(X_pp_leg, X_pp_soa, "X_pp Intermediate Construction")


def test_04_v_ppph_contribution(dummy_data):
    """
    COMPONENT CHECK: Verify the contribution of v_ppph to H1 and X_pp.
    This checks the tricky if/else sparse logic logic and signs.
    """
    d = dummy_data
    
    
    term_H1_leg = -0.5 * contract('cdak, cdki -> ai', d['v_ppph'], d['t2'])
    term_Xpp_leg = -contract('cdak, ck -> ad', d['v_ppph'], d['t1'])
    
    t1 = to_torch(d['t1'])
    t2 = to_torch(d['t2'])
    v_ppph_dense = to_torch(d['v_ppph'])
    
    indices = torch.nonzero(v_ppph_dense).T
    values = v_ppph_dense[indices[0], indices[1], indices[2], indices[3]]
    v_ppph_soa = (indices, values)
    
    v_ppph_results = soa_dgrams.v_ppph_dgrams(v_ppph_soa, t1, t2)
    
    # H1 += v_ppph_results[0]
    # X_pp += v_ppph_results[1]
    
    term_H1_soa = v_ppph_results[0]
    term_Xpp_soa = v_ppph_results[1]
    
    assert_tensors_match(term_H1_leg, term_H1_soa, "v_ppph contribution to H1")
    assert_tensors_match(term_Xpp_leg, term_Xpp_soa, "v_ppph contribution to X_pp")


def test_05_denominator_consistency(dummy_data):
    """
    COMPONENT CHECK: Verify denominators match.
    Legacy: denom = - np.add.outer(diag_p, diag_h) where diag_p/h come from X
    """
    d = dummy_data
    
    X_pp_leg = np.random.randn(*d['f_pp'].shape)
    X_hh_leg = np.random.randn(*d['f_hh'].shape)
    
    diag_p_leg = np.diag(X_pp_leg)
    diag_h_leg = np.diag(X_hh_leg)
    denom_leg = - np.add.outer(diag_p_leg, diag_h_leg)
    
    X_pp_soa = to_torch(X_pp_leg)
    X_hh_soa = to_torch(X_hh_leg)
    
    diag_p_soa = torch.diagonal(X_pp_soa)
    diag_h_soa = torch.diagonal(X_hh_soa)
    denom_soa = -(diag_p_soa.unsqueeze(1) + diag_h_soa.unsqueeze(0))
    
    assert_tensors_match(denom_leg, denom_soa, "Denominator Construction")


def test_06_full_t1iter_step_fixed(dummy_data):
    """
    INTEGRATION CHECK: Full t1Iter legacy vs SoA.
    Uses deep copies to prevent legacy in-place mutation from poisoning SoA inputs.
    """
    d = dummy_data
    
    t1_orig = d['t1'].copy()
    t2_orig = d['t2'].copy()
    f_ph_orig = d['f_ph'].copy()
    f_pp_orig = d['f_pp'].copy()
    f_hh_orig = d['f_hh'].copy()
    v_phph_orig = d['v_phph'].copy()
    v_phhh_orig = d['v_phhh'].copy()
    v_pphh_orig = d['v_pphh'].copy()
    v_ppph_orig = d['v_ppph'].copy()

    print("\n--- Running Legacy ---")
    print("Value before legacy:", d['f_ph'][0,0])
    t1_next_leg = legacy_cc.t1Iter(
        d['t1'], d['t2'], d['f_ph'], d['f_pp'], d['f_hh'],
        d['v_phph'], d['v_phhh'], d['v_pphh'], d['v_ppph'], 
        sparse=False
    )
    print("Value after legacy:", d['f_ph'][0,0])
    print("--- Running SoA ---")
    t1_t = to_torch(t1_orig)
    t2_t = to_torch(t2_orig)
    f_ph_t = to_torch(f_ph_orig)
    f_pp_t = to_torch(f_pp_orig)
    f_hh_t = to_torch(f_hh_orig)
    
    v_phph_t = to_torch(v_phph_orig)
    v_phhh_t = to_torch(v_phhh_orig)
    v_pphh_t = to_torch(v_pphh_orig)
    v_ppph_t = to_torch(v_ppph_orig)

    t1_next_soa_dense = soa_cc.t1Iter(
        t1_t, t2_t, f_ph_t, f_pp_t, f_hh_t, 
        v_phph_t, v_phhh_t, v_pphh_t, v_ppph_t, 
        sparse=False
    )
    
    assert_tensors_match(
        t1_next_leg, 
        to_numpy(t1_next_soa_dense), 
        "Full T1 Update (Dense Path)"
    )


@pytest.fixture
def probe_data():
    """Generates consistent physics data for internal probing."""
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

def test_probe_t1iter_internals(probe_data):
    """
    Probes the internal components of T1 iteration to find the 1e-2 drift.
    """
    d = probe_data
    t1_t, t2_t = torch.tensor(d['t1']), torch.tensor(d['t2'])
    f_ph_t, f_pp_t, f_hh_t = torch.tensor(d['f_ph']), torch.tensor(d['f_pp']), torch.tensor(d['f_hh'])
    v_phph_t, v_phhh_t, v_pphh_t, v_ppph_t = torch.tensor(d['v_phph']), torch.tensor(d['v_phhh']), torch.tensor(d['v_pphh']), torch.tensor(d['v_ppph'])

    h1_leg = d['f_ph'].copy()
    h1_leg += legacy_dgrams.dgram_akci_ck(d['v_phph'], d['t1'])
    h1_leg += legacy_dgrams.dgram_ck_acik(d['f_ph'], d['t2'])
    h1_leg += legacy_dgrams.dgram_cikl_cakl(d['v_phhh'], d['t2'])
    h1_leg += legacy_dgrams.dgram_cdkl_ck_dali(d['v_pphh'], d['t1'], d['t2'])

    h1_soa = f_ph_t.clone()
    h1_soa += soa_dgrams.dgram_akci_ck(v_phph_t, t1_t)
    h1_soa += soa_dgrams.dgram_ck_acik(f_ph_t, t2_t)
    h1_soa += soa_dgrams.dgram_cikl_cakl(v_phhh_t, t2_t)
    h1_soa += soa_dgrams.dgram_cdkl_ck_dali(v_pphh_t, t1_t, t2_t)

    assert np.allclose(h1_leg, to_numpy(h1_soa), atol=1e-12), "MISMATCH: Base Numerator Diagrams (H1)"

    x_pp_leg = d['f_pp'].copy()
    x_pp_leg += legacy_dgrams.dgram_ck_ak(d['f_ph'], d['t1'])
    x_pp_leg += legacy_dgrams.dgram_dckl_dakl(d['v_pphh'], d['t2'])
    x_pp_leg += legacy_dgrams.dgram_cdkl_dk_al(d['v_pphh'], d['t1'])
    x_pp_leg -= contract('cdak, ck -> ad', d['v_ppph'], d['t1'])

    x_pp_soa = f_pp_t.clone()
    x_pp_soa += soa_dgrams.dgram_ck_ak(f_ph_t, t1_t)
    x_pp_soa += soa_dgrams.dgram_dckl_dakl(v_pphh_t, t2_t)
    x_pp_soa += soa_dgrams.dgram_cdkl_dk_al(v_pphh_t, t1_t)
    x_pp_soa -= torch.einsum('cdak, ck -> ad', v_ppph_t, t1_t)

    assert np.allclose(x_pp_leg, to_numpy(x_pp_soa), atol=1e-12), "MISMATCH: Intermediate Particle Hamiltonian (X_pp)"

    term_linear_leg = contract('ac, ci -> ai', x_pp_leg, d['t1'])
    term_linear_soa = torch.einsum("ac, ci -> ai", x_pp_soa, t1_t)

    assert np.allclose(term_linear_leg, to_numpy(term_linear_soa), atol=1e-12), "MISMATCH: Final X_pp * T1 contraction"

    diag_h_leg = np.diag(-d['f_hh']) 
    diag_p_leg = np.diag(x_pp_leg)
    denom_leg = - np.add.outer(diag_p_leg, diag_h_leg)

    diag_h_soa = torch.diagonal(-f_hh_t)
    diag_p_soa = torch.diagonal(x_pp_soa)
    denom_soa = -(diag_p_soa.unsqueeze(1) + diag_h_soa.unsqueeze(0))

    assert np.allclose(denom_leg, to_numpy(denom_soa), atol=1e-12), "MISMATCH: Energy Denominators"

def test_t1_component_auditor(dummy_data):
    d = dummy_data

    t1, t2 = to_torch(d['t1']), to_torch(d['t2'])
    f_ph, f_pp, f_hh = to_torch(d['f_ph']), to_torch(d['f_pp']), to_torch(d['f_hh'])
    
    soa_H1 = f_ph.clone()
    leg_H1 = d['f_ph'].copy()
    assert np.allclose(leg_H1, to_numpy(soa_H1)), "Fock ph mismatch"

    soa_term = soa_dgrams.dgram_cdkl_ck_dali(to_torch(d['v_pphh']), t1, t2)
    leg_term = legacy_dgrams.dgram_cdkl_ck_dali(d['v_pphh'], d['t1'], d['t2'])
    
    assert np.allclose(leg_term, to_numpy(soa_term)), "N^6 Diagram (v_pphh * t1 * t2) mismatch"

    soa_Xpp = f_pp.clone()
    soa_Xpp += soa_dgrams.dgram_ck_ak(f_ph, t1)
    
    leg_Xpp = d['f_pp'].copy()
    leg_Xpp += legacy_dgrams.dgram_ck_ak(d['f_ph'], d['t1'])
    
    assert np.allclose(leg_Xpp, to_numpy(soa_Xpp)), "X_pp dressing mismatch"
