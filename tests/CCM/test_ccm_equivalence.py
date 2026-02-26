import pytest
import torch
import NuLattice.lattice as lat
import NuLattice.references as ref
import NuLattice.CCM.coupled_cluster as old_ccm
import NuLattice.CCM.soa.coupled_cluster as new_ccm

LATTICE_SIZE = 2
A_LAT = 2.5
VT1 = -9.0
VS1 = -9.0
W3 = 6.0
REF_STATE = ref.ref_16O_gs


@pytest.fixture(scope="module")
def shared_params():
    """Returns the common physics parameters for benchmarks."""
    return {
        "thisL": LATTICE_SIZE,
        "ref_state": REF_STATE,
        "vT1": VT1,
        "vS1": VS1,
        "w3": W3,
        "a_lat": A_LAT,
    }


def test_ccsd_numerical_parity(shared_params):
    """
    Standard pytest to ensure the new SoA implementation
    matches legacy within an acceptable tolerance.
    """
    p = shared_params

    refEn_old, fock_old, v2b_old = old_ccm.get_norm_ord_int(
        p["thisL"], p["ref_state"], p["vT1"], p["vS1"], p["w3"], sparse=True
    )
    corr_old, _, _ = old_ccm.ccsd_solver(
        fock_old, v2b_old, eps=1e-8, maxSteps=500, max_diis=10, sparse=True
    )
    e_old = (corr_old + refEn_old) * lat.phys_unit(p["a_lat"])

    refEn_new, fock_new, v2b_new = new_ccm.get_norm_ord_int(
        p["thisL"], p["ref_state"], p["vT1"], p["vS1"], p["w3"], sparse=True
    )
    corr_new, _, _ = new_ccm.ccsd_solver(
        fock_new,
        v2b_new,
        eps=1e-8,
        maxSteps=500,
        max_diis=10,
        sparse=True,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    e_new = (corr_new + refEn_new) * lat.phys_unit(p["a_lat"])

    diff = abs(e_old - e_new)
    assert diff < 1e-5, f"Numerical divergence detected: {diff:.2e} MeV"


@pytest.mark.benchmark(group="ccsd_solver")
def test_benchmark_legacy_solver(shared_params, benchmark):
    p = shared_params
    refEn, fock, v2b = old_ccm.get_norm_ord_int(
        p["thisL"], p["ref_state"], p["vT1"], p["vS1"], p["w3"], sparse=True
    )

    def run_solver():
        return old_ccm.ccsd_solver(
            fock, v2b, eps=1e-8, maxSteps=100, max_diis=10, sparse=True
        )

    benchmark(run_solver)


@pytest.mark.benchmark(group="ccsd_solver")
def test_benchmark_soa_solver(shared_params, benchmark):
    p = shared_params
    refEn, fock, v2b = new_ccm.get_norm_ord_int(
        p["thisL"], p["ref_state"], p["vT1"], p["vS1"], p["w3"], sparse=True
    )
    new_ccm.ccsd_solver(
        fock,
        v2b,
        eps=1e-8,
        maxSteps=100,
        max_diis=10,
        sparse=True,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    def run_solver():
        return new_ccm.ccsd_solver(
            fock,
            v2b,
            eps=1e-8,
            maxSteps=100,
            max_diis=10,
            sparse=True,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

    benchmark(run_solver)


def test_profile_soa_memory(shared_params):
    """
    Isolated node for Memray profiling.
    Run via: memray run -m pytest <file.py>::test_profile_soa_memory
    """
    p = shared_params
    refEn, fock, v2b = new_ccm.get_norm_ord_int(
        p["thisL"], p["ref_state"], p["vT1"], p["vS1"], p["w3"], sparse=True
    )
    new_ccm.ccsd_solver(
        fock,
        v2b,
        eps=1e-8,
        maxSteps=50,
        max_diis=10,
        sparse=True,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
