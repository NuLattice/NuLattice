import pytest
import numpy as np
import torch
import NuLattice.lattice as lat
import NuLattice.references as ref

import NuLattice.CCM.coupled_cluster as legacy_cc
import NuLattice.CCM.three_body_utils as legacy_tbu
import NuLattice.CCM.soa.coupled_cluster as soa_cc
import NuLattice.CCM.soa.three_body_utils as soa_tbu
import NuLattice.CCM.soa.ccDgrams as soa_dgrams


def to_numpy(t):
    """Converts Torch tensors (or Operators) to NumPy for comparison."""
    if hasattr(t, "to_numpy"):
        return t.to_numpy()
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


def assert_tensors_match(legacy, new, name, tol=1e-7):
    arr_new = to_numpy(new)

    assert legacy.shape == arr_new.shape, (
        f"Shape mismatch in {name}: {legacy.shape} vs {arr_new.shape}"
    )

    if not np.allclose(legacy, arr_new, atol=tol, rtol=tol):
        diff = np.abs(legacy - arr_new)
        max_diff = np.max(diff)
        print(f"\n[FAIL] {name} max diff: {max_diff:.2e}")

        mismatch_idx = np.where(diff > tol)
        print(f"Indices of mismatch (first 3): {list(zip(*mismatch_idx))[:3]}")
        print(f"Legacy vals: {legacy[mismatch_idx][:3]}")
        print(f"New vals   : {arr_new[mismatch_idx][:3]}")

        pytest.fail(f"{name} mismatch")


@pytest.fixture
def physics_inputs():
    """Exact parameters from your diverging benchmark."""
    return {
        "thisL": 2,
        "vT1": -9.0,
        "vS1": -9.0,
        "w3": 6.0,
        "holes": ref.ref_16O_gs,
        "sparse": True,
    }


def test_01_hamiltonian_setup(physics_inputs):
    """
    Did we build the correct matrices?
    Checks: get_norm_ord_int -> get_all_interactions, get_fock_matrices, get_3NF
    """
    p = physics_inputs

    vac_old, f_old, v_old = legacy_cc.get_norm_ord_int(
        p["thisL"],
        p["holes"],
        p["vT1"],
        p["vS1"],
        p["w3"],
        sparse=False,
    )

    vac_new, f_new, v_new = soa_cc.get_norm_ord_int(
        p["thisL"], p["holes"], p["vT1"], p["vS1"], p["w3"], sparse=False
    )

    print(f"\nReference Energy: Legacy={vac_old:.6f}, New={vac_new:.6f}")
    assert np.isclose(vac_old, vac_new), "Reference Energy Mismatch!"

    f_names = ["f_pp", "f_ph", "f_hh"]
    for i, name in enumerate(f_names):
        assert_tensors_match(f_old[i], f_new[i], name)

    v_names = ["v_pppp", "v_ppph", "v_pphh", "v_phph", "v_phhh", "v_hhhh"]
    for i, name in enumerate(v_names):
        assert_tensors_match(v_old[i], v_new[i], name)


def test_02_initialization(physics_inputs):
    """
    Do we start from the same guess?
    Checks: t1Init, t2Init (Denominator logic)
    """
    p = physics_inputs
    _, fock, two_body = legacy_cc.get_norm_ord_int(
        p["thisL"], p["holes"], p["vT1"], p["vS1"], p["w3"], sparse=False
    )
    f_pp, f_ph, f_hh = fock
    v_pphh = two_body[2]

    t1_old = legacy_cc.t1Init(f_ph, f_pp, f_hh, delta=0)
    t2_old = legacy_cc.t2Init(f_pp, f_hh, v_pphh, delta=0)

    t_f_pp = torch.tensor(f_pp)
    t_f_ph = torch.tensor(f_ph)
    t_f_hh = torch.tensor(f_hh)
    t_v_pphh = torch.tensor(v_pphh)

    t1_new = soa_cc.t1Init(t_f_ph, t_f_pp, t_f_hh, delta=0)
    t2_new = soa_cc.t2Init(t_f_pp, t_f_hh, t_v_pphh, delta=0)

    assert_tensors_match(t1_old, t1_new, "Initial T1")
    assert_tensors_match(t2_old, t2_new, "Initial T2")


def test_03_sparse_kernel_math():
    """
    Do the manually flattened scatter-adds match the tensor math?
    Checks: soa/ccDgrams.py vs legacy einsums
    """

    p, h = 4, 4
    torch.manual_seed(42)

    t1 = torch.randn(p, h, dtype=torch.float64)
    t2 = torch.randn(p, p, h, h, dtype=torch.float64)

    v_ppph_dense = torch.randn(p, p, p, h, dtype=torch.float64)

    mask = torch.rand_like(v_ppph_dense) > 0.5
    v_ppph_dense[mask] = 0.0

    indices = torch.nonzero(v_ppph_dense).T  # (4, nnz)
    values = v_ppph_dense[indices[0], indices[1], indices[2], indices[3]]
    v_ppph_soa = (indices, values)

    res_new = soa_dgrams.v_ppph_dgrams(v_ppph_soa, t1, t2)

    # H1 += - 0.5 * contract('cdak, cdki -> ai', v, t2)
    ref_0 = -0.5 * torch.einsum("cdak, cdki -> ai", v_ppph_dense, t2)
    assert_tensors_match(ref_0.numpy(), res_new[0], "v_ppph dgram 0 (H1 contribution)")

    # X_pp -= contract('cdak, ck -> ad', v, t1)
    # ret1 (updates X_pp) and ret3 (updates X_pp transpose?)
    ref_1 = -torch.einsum("cdak, ck -> ad", v_ppph_dense, t1)
    # ret1[a, d] -= v * t1[c, k] -> matches sign?
    assert_tensors_match(
        ref_1.numpy(), res_new[1], "v_ppph dgram 1 (X_pp contribution)"
    )

    # H2 += pIJ(contract('abcj, ci -> abij', v, t1))
    # v_ppph is [a,b,c,j]. New kernel inputs [c,d,a,k]. Mapping required.
    # ret2[c, d, j, k] += v[c,d,a,k] * t1[a, j]
    # v indices: 0,1,2,3 -> c,d,a,k.
    # contraction: sum_a v[c,d,a,k] * t1[a,j] -> res[c,d,j,k]
    ref_2 = torch.einsum("cdak, aj -> cdjk", v_ppph_dense, t1)
    assert_tensors_match(ref_2.numpy(), res_new[2], "v_ppph dgram 2")


def test_04_solver_step_execution(physics_inputs):
    """
    Does one iteration produce the same result?
    Checks: t1Iter, t2Iter integration
    """
    p = physics_inputs

    _, f_old, v_old = legacy_cc.get_norm_ord_int(
        p["thisL"], p["holes"], p["vT1"], p["vS1"], p["w3"], sparse=False
    )

    t1 = legacy_cc.t1Init(f_old[1], f_old[0], f_old[2], 0)
    t2 = legacy_cc.t2Init(f_old[0], f_old[2], v_old[2], 0)

    t1_next_old = legacy_cc.t1Iter(
        t1,
        t2,
        f_old[1],
        f_old[0],
        f_old[2],
        v_old[3],
        v_old[4],
        v_old[2],
        v_old[1],
        sparse=False,
    )

    t1_t = torch.tensor(t1)
    t2_t = torch.tensor(t2)
    f_t = [torch.tensor(x, dtype=torch.float64) for x in f_old]
    v_t = [torch.tensor(x, dtype=torch.float64) for x in v_old]

    t1_next_new = soa_cc.t1Iter(
        t1_t,
        t2_t,
        f_t[1],
        f_t[0],
        f_t[2],
        v_t[3],
        v_t[4],
        v_t[2],
        v_t[1],
        sparse=False,
    )

    assert_tensors_match(t1_next_old, t1_next_new, "T1 Iteration Result")


def test_debug_fock_traces(physics_inputs):
    """Isolate the trace logic: f = h + sum(V)."""
    p = physics_inputs
    lattice = lat.get_lattice(p["thisL"])
    myTkin = lat.Tkin(lattice, p["thisL"])
    hole, part = lat.states2PHSpace(p["holes"], p["thisL"])

    v_old = legacy_cc.get_all_interactions(
        part, hole, lat.contacts(-9.0, -9.0, lattice, p["thisL"]), sparse=False
    )
    v_phph_old, v_phhh_old, v_hhhh_old = v_old[3], v_old[4], v_old[5]

    f_old = legacy_cc.get_fock_matrices(
        part, hole, myTkin, v_phph_old, v_phhh_old, v_hhhh_old
    )

    v_phph_t = torch.tensor(v_phph_old)
    v_phhh_t = torch.tensor(v_phhh_old)
    v_hhhh_t = torch.tensor(v_hhhh_old)

    f_new = soa_cc.get_fock_matrices(part, hole, myTkin, v_phph_t, v_phhh_t, v_hhhh_t)

    assert_tensors_match(f_old[0], f_new[0], "Fock f_pp (Trace Check)")
    assert_tensors_match(f_old[2], f_new[2], "Fock f_hh (Trace Check)")


def test_debug_3nf_to_2b_scaling(physics_inputs):
    """Verify if the 3NF -> NO2B contribution has the correct scaling/factors."""
    p = physics_inputs
    lattice = lat.get_lattice(p["thisL"])
    hole, part = lat.states2PHSpace(p["holes"], p["thisL"])
    my3body = lat.NNNcontact(p["w3"], lattice, p["thisL"])

    res_old = legacy_tbu.get_3NF(part, hole, my3body)
    dum_2b_old = legacy_tbu.get_3NF_tbme(
        res_old[2],
        res_old[4],
        res_old[5],
        res_old[6],
        res_old[7],
        res_old[8],
        len(part),
        len(hole),
        sparse_pppp=False,
        sparse_ppph=False,
    )

    res_soa = soa_tbu.get_3NF(part, hole, my3body)
    dum_2b_new = soa_tbu.get_3NF_tbme(
        res_soa[2],
        res_soa[4],
        res_soa[5],
        res_soa[6],
        res_soa[7],
        res_soa[8],
        len(part),
        len(hole),
        sparse_pppp=False,
        sparse_ppph=False,
    )

    assert_tensors_match(dum_2b_old[3], dum_2b_new[3], "3NF -> NO2B (v_phph)")
    assert_tensors_match(dum_2b_old[0], dum_2b_new[0], "3NF -> NO2B (v_pppp)")


# NOTE(vivek): Fails beyond 1e-7 tolerance
def test_debug_sparse_vs_dense_t1iter():
    """Does t1Iter produce the same result using Sparse Kernels vs Dense Einsums?"""
    p, h = 8, 4
    torch.manual_seed(123)

    f_ph = torch.randn(p, h)
    f_pp = torch.diag(torch.randn(p) + 10)
    f_hh = torch.diag(torch.randn(h) - 10)
    t1 = torch.randn(p, h) * 0.1
    t2 = torch.randn(p, p, h, h) * 0.1

    v_phph = torch.randn(p, h, p, h)
    v_phhh = torch.randn(p, h, h, h)
    v_pphh = torch.randn(p, p, h, h)
    v_ppph_dense = torch.randn(p, p, p, h)

    t1_dense = soa_cc.t1Iter(
        t1, t2, f_ph, f_pp, f_hh, v_phph, v_phhh, v_pphh, v_ppph_dense, sparse=False
    )

    indices = torch.nonzero(v_ppph_dense).T
    values = v_ppph_dense[indices[0], indices[1], indices[2], indices[3]]
    v_ppph_soa = (indices, values)

    v_ppph_res = soa_dgrams.v_ppph_dgrams(v_ppph_soa, t1, t2)
    t1_sparse = soa_cc.t1Iter(
        t1, t2, f_ph, f_pp, f_hh, v_phph, v_phhh, v_pphh, v_ppph_res, sparse=True
    )

    assert_tensors_match(
        t1_dense.numpy(), t1_sparse.numpy(), "T1Iter Sparse-Dense Consistency"
    )


def test_debug_single_element_consistency():
    """Isolate a single V element to see where it lands in T1."""
    p, h = 2, 2
    t2 = torch.zeros((p, p, h, h), dtype=torch.float64)
    t2[0, 1, 0, 0] = 1.0
    t1 = torch.zeros((p, h), dtype=torch.float64)

    # c=0, d=1, a=0, k=0
    indices = torch.tensor([[0], [1], [0], [0]], dtype=torch.long)
    values = torch.tensor([1.0], dtype=torch.float64)
    soa = (indices, values)

    v_dense = torch.zeros((p, p, p, h), dtype=torch.float64)
    v_dense[0, 1, 0, 0] = 1.0

    res_sparse = soa_dgrams.v_ppph_dgrams(soa, t1, t2)
    res_dense = -0.5 * torch.einsum("cdak, cdki -> ai", v_dense, t2)

    print(f"\nSparse Result (ret0):\n{res_sparse[0]}")
    print(f"Dense Result:\n{res_dense}")

    assert torch.allclose(res_sparse[0], res_dense)


def test_debug_hamiltonian_mismatch(physics_inputs):
    p = physics_inputs

    vac_old, f_old, v_old = legacy_cc.get_norm_ord_int(
        p["thisL"], p["holes"], p["vT1"], p["vS1"], p["w3"], sparse=False
    )

    vac_new, f_new, v_new = soa_cc.get_norm_ord_int(
        p["thisL"], p["holes"], p["vT1"], p["vS1"], p["w3"], sparse=False
    )

    print(f"\nRef Energy Diff: {abs(vac_old - vac_new):.2e}")

    print(
        f"f_pp diag diff: {np.abs(np.diag(f_old[0]) - np.diag(to_numpy(f_new[0]))).max():.2e}"
    )
    print(
        f"f_hh diag diff: {np.abs(np.diag(f_old[2]) - np.diag(to_numpy(f_new[2]))).max():.2e}"
    )

    assert_tensors_match(v_old[2], v_new[2], "v_pphh")


def test_debug_solver_path_consistency(physics_inputs):
    """Compare 10 steps of the full solver logic."""
    p = physics_inputs

    vac_old, f_old, v_old = legacy_cc.get_norm_ord_int(
        p["thisL"], p["holes"], p["vT1"], p["vS1"], p["w3"], sparse=False
    )
    vac_new, f_new, v_new = soa_cc.get_norm_ord_int(
        p["thisL"], p["holes"], p["vT1"], p["vS1"], p["w3"], sparse=False
    )

    e_old, t1_old, t2_old = legacy_cc.ccsd_solver(
        f_old, v_old, maxSteps=10, max_diis=0, mixing=0, verbose=False, sparse=False
    )

    e_new, t1_new, t2_new = soa_cc.ccsd_solver(
        f_new, v_new, maxSteps=10, max_diis=0, mixing=0, verbose=False, sparse=False
    )

    print(f"\nEnergy after 10 steps: Legacy={e_old:.6f}, New={e_new:.6f}")
    assert np.isclose(e_old, e_new, atol=1e-8)


def test_debug_multi_element_collision():
    """Verify that multiple sparse elements correctly accumulate (sum) in ret0."""
    p, h = 2, 2
    t2 = torch.zeros((p, p, h, h), dtype=torch.float64)

    # Set t2 values for two different (c,d,k) combinations
    t2[0, 1, 0, 0] = 1.0
    t2[1, 0, 1, 0] = 1.0
    t1 = torch.zeros((p, h), dtype=torch.float64)

    # Indices: [c, d, a, k]
    indices = torch.tensor([[0, 1], [1, 0], [0, 0], [0, 1]], dtype=torch.long)
    values = torch.tensor([1.0, 1.0], dtype=torch.float64)
    soa = (indices, values)

    res_sparse = soa_dgrams.v_ppph_dgrams(soa, t1, t2)

    # Manually calculated:
    # Term 1: -0.5 * 1.0 * t2[0,1,0,0] = -0.5
    # Term 2: -0.5 * 1.0 * t2[1,0,1,0] = -0.5
    # Total should be -1.0 at [0,0]
    expected = -1.0
    actual = res_sparse[0][0, 0].item()

    assert np.isclose(actual, expected), (
        f"Collision Failed: Expected {expected}, got {actual}"
    )


def test_debug_flattened_stride_overlap():
    """Verify that scattering into ret2 does not leak into adjacent memory."""
    p, h = 4, 4
    t1 = torch.zeros((p, h), dtype=torch.float64)
    t1[0, 0] = 1.0
    t1[1, 1] = 2.0
    t2 = torch.zeros((p, p, h, h))

    # Two elements with different 'a' and 'j'
    # V[c,d,a,k] -> V[0,0,0,0]=1.0 and V[0,0,1,1]=1.0
    indices = torch.tensor([[0, 0], [0, 0], [0, 1], [0, 1]], dtype=torch.long)
    values = torch.tensor([1.0, 1.0], dtype=torch.float64)
    soa = (indices, values)

    res_sparse = soa_dgrams.v_ppph_dgrams(soa, t1, t2)

    # ret2[c, d, j, k] += V[c,d,a,k] * T1[a, j]
    # ret2[0, 0, 0, 0] += V[0,0,0,0]*T1[0,0] = 1*1 = 1
    # ret2[0, 0, 1, 1] += V[0,0,1,1]*T1[1,1] = 1*2 = 2

    assert res_sparse[2][0, 0, 0, 0] == 1.0
    assert res_sparse[2][0, 0, 1, 1] == 2.0
    assert torch.sum(res_sparse[2]) == 3.0


def test_04_debug_initial_amplitudes(physics_inputs):
    p = physics_inputs
    _, f_old, v_old = legacy_cc.get_norm_ord_int(
        p["thisL"], p["holes"], p["vT1"], p["vS1"], p["w3"], sparse=False
    )

    t1_old = legacy_cc.t1Init(f_old[1], f_old[0], f_old[2], delta=0)
    t1_new = soa_cc.t1Init(
        torch.tensor(f_old[1]), torch.tensor(f_old[0]), torch.tensor(f_old[2]), delta=0
    )
    assert_tensors_match(t1_old, t1_new, "Init T1 Comparison")

    t2_old = legacy_cc.t2Init(f_old[0], f_old[2], v_old[2], delta=0)
    t2_new = soa_cc.t2Init(
        torch.tensor(f_old[0]), torch.tensor(f_old[2]), torch.tensor(v_old[2]), delta=0
    )
    assert_tensors_match(t2_old, t2_new, "Init T2 Comparison")


def test_debug_t1_numerator_isolation(physics_inputs):
    """
    Identifies if the mismatch is in the Numerator (Diagram Sums)
    or the Denominator (Single Particle Energies).
    """
    p = physics_inputs
    _, f_old, v_old = legacy_cc.get_norm_ord_int(
        p["thisL"], p["holes"], p["vT1"], p["vS1"], p["w3"], sparse=False
    )

    t1 = legacy_cc.t1Init(f_old[1], f_old[0], f_old[2], 0)
    t2 = legacy_cc.t2Init(f_old[0], f_old[2], v_old[2], 0)

    # H1_total = f_ai + diagrams + linear_terms
    # Shift = t1_next - t1_old = H1_total / Denom

    t1_next_old = legacy_cc.t1Iter(
        t1,
        t2,
        f_old[1],
        f_old[0],
        f_old[2],
        v_old[3],
        v_old[4],
        v_old[2],
        v_old[1],
        sparse=False,
    )

    # Reconstruct legacy dressed diagonals for denominator
    # (X_hh = -f_hh + dressing, X_pp = f_pp + dressing)

    t1_t = torch.tensor(t1)
    t2_t = torch.tensor(t2)
    f_t = [torch.tensor(x) for x in f_old]

    indices = torch.nonzero(torch.tensor(v_old[1])).T
    values = torch.tensor(v_old[1])[indices[0], indices[1], indices[2], indices[3]]
    v_ppph_res = soa_dgrams.v_ppph_dgrams((indices, values), t1_t, t2_t)

    t1_next_new = soa_cc.t1Iter(
        t1_t,
        t2_t,
        f_t[1],
        f_t[0],
        f_t[2],
        torch.tensor(v_old[3]),
        torch.tensor(v_old[4]),
        torch.tensor(v_old[2]),
        v_ppph_res,
        sparse=True,
    )

    legacy_shift = t1_next_old - t1
    new_shift = to_numpy(t1_next_new) - t1

    print(f"\nMax shift difference: {np.abs(legacy_shift - new_shift).max():.2e}")


def test_05_t2iter_step(physics_inputs):
    """
    INTEGRATION CHECK: Does T2 iteration match legacy exactly?
    """
    p = physics_inputs
    _, f_old, v_old = legacy_cc.get_norm_ord_int(
        p["thisL"], p["holes"], p["vT1"], p["vS1"], p["w3"], sparse=False
    )

    t1_init = legacy_cc.t1Init(f_old[1], f_old[0], f_old[2], 0)
    t2_init = legacy_cc.t2Init(f_old[0], f_old[2], v_old[2], 0)

    t1_t = torch.tensor(t1_init)
    t2_t = torch.tensor(t2_init)
    f_t = [torch.tensor(x) for x in f_old]
    v_t = [torch.tensor(x) for x in v_old]

    t2_next_leg = legacy_cc.t2Iter(
        t1_init.copy(),
        t2_init.copy(),
        f_old[1],
        f_old[2],
        f_old[0],
        v_old[0],
        v_old[3],
        v_old[4],
        v_old[2],
        v_old[1],
        v_old[5],
        sparse=False,
    )

    t2_next_soa = soa_cc.t2Iter(
        t1_t,
        t2_t,
        f_t[1],
        f_t[2],
        f_t[0],
        v_t[0],
        v_t[3],
        v_t[4],
        v_t[2],
        v_t[1],
        v_t[5],
        sparse=False,
    )

    assert_tensors_match(t2_next_leg, to_numpy(t2_next_soa), "T2 Iteration Result")
