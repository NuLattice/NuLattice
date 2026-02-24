import numpy as np
import pytest
import NuLattice.lattice_soa as lat
import NuLattice.references as ref


@pytest.fixture
def imsrg_setup():
    """Pre-calculates operators and occupations for Normal Ordering benchmarks."""
    thisL = 6  # Basis size N=864

    my_basis = lat.get_sp_basis(thisL)
    n_states = len(my_basis)
    lattice = lat.get_lattice(thisL)

    # Generate SoA Operators
    myTkin = lat._Tkin(lattice, thisL)
    mycontact = lat._contacts(-9.0, -9.0, lattice, thisL)
    my3body = lat._NNNcontact(6.0, lattice, thisL)

    # Setup 16O ground state and occupations
    my_ref = ref.ref_16O_gs
    # Assuming NuLattice.IMSRG contains the create_occupations utility
    from NuLattice.IMSRG.normal_ordering import create_occupations

    occs = create_occupations(my_basis, my_ref)

    
    setup_data = {"occs": occs, "ops": (myTkin, mycontact, my3body), "n_states": n_states}

    import NuLattice.IMSRG.soa.normal_ordering as no_soa
    
    _ = no_soa.compute_normal_ordered_hamiltonian_no2b(
        setup_data["occs"], *setup_data["ops"]
    )
    return setup_data

@pytest.mark.benchmark(group="IMSRG_NormalOrdering")
def test_normal_ordering_aos(benchmark, imsrg_setup):
    """Benchmarks the legacy AoS implementation of normal ordering."""
    import NuLattice.IMSRG.normal_ordering as no

    occs = imsrg_setup["occs"]
    op1, op2, op3 = (op.to_list() for op in imsrg_setup["ops"])

    def run_normal_ordering():
        e0, f, gamma = no.compute_normal_ordered_hamiltonian_no2b(occs, op1, op2)
        return e0

    result_e0 = benchmark(run_normal_ordering)
    assert result_e0 is not None


@pytest.mark.benchmark(group="IMSRG_NormalOrdering")
def test_normal_ordering_soa(benchmark, imsrg_setup):
    """Benchmarks the optimized SoA + Numba implementation of normal ordering."""
    import NuLattice.IMSRG.soa.normal_ordering as no

    occs = imsrg_setup["occs"]
    ops = imsrg_setup["ops"]

    

    def run_normal_ordering():
        e0, f, gamma = no.compute_normal_ordered_hamiltonian_no2b(occs, ops[0], ops[1])
        return e0

    result_e0 = benchmark(run_normal_ordering)
    assert result_e0 is not None

@pytest.mark.benchmark(group="IMSRG_Compute_Only")
def test_soa_compute_only(benchmark, imsrg_setup):
    from NuLattice.IMSRG.soa.normal_ordering import (
        _compute_op1_kernel, _compute_op2_kernel, _compute_op3_kernel
    )
    
    dim = imsrg_setup["n_states"]
    e0_arr = np.zeros(1)
    f = np.zeros((dim, dim))
    gamma = np.zeros((dim, dim, dim, dim)) # primary memory and time cost
    
    op1, op2, op3 = imsrg_setup["ops"]
    h1_mat = op1.to_dense()

    def run_kernels():
        e0_arr[0] = 0
        f.fill(0)
        
        _compute_op1_kernel(h1_mat, dim, imsrg_setup["occs"], e0_arr, f)
        _compute_op2_kernel(op2.indices, op2.values, imsrg_setup["occs"], e0_arr, f, gamma)
        if op3:
            _compute_op3_kernel(op3.indices, op3.values, imsrg_setup["occs"], e0_arr, f, gamma)

    benchmark(run_kernels)

@pytest.mark.benchmark(group="IMSRG_Compute_Only")
def test_aos_compute_only(benchmark, imsrg_setup):
    """
    Benchmarks the raw Python loop performance of the AoS implementation
    by removing OS allocation and list expansion overhead.
    """
    import NuLattice.IMSRG.normal_ordering as no_aos
    occs = imsrg_setup["occs"]
    dim = len(occs)
    
    op1, op2, op3 = imsrg_setup["ops"]
    h1_list = op1.to_list()
    h2_list = op2.to_list()
    h3_list = op3.to_list() if op3 else None

    h2_expanded = no_aos.expand_h2(h2_list)
    h3_expanded = no_aos.expand_h3(h3_list) if h3_list else None
    f_buffer = np.zeros((dim, dim))
    gamma_buffer = np.zeros((dim, dim, dim, dim))

    def run_legacy_loops():
        e0 = 0.0
        f_buffer.fill(0)
        
        for p, q, me in h1_list:
            if p == q:
                e0 += occs[p] * me
            f_buffer[p, q] += me

        for p, q, r, s, me in h2_expanded:
            if q == s:
                if p == r:
                    e0 += occs[p] * occs[q] * 0.5 * me
                f_buffer[p, r] += occs[q] * me
            gamma_buffer[p, q, r, s] += me

        if h3_expanded is not None:
            for p, q, r, s, t, u, me in h3_expanded:
                if r == u:
                    if q == t:
                        if p == s:
                            e0 += occs[p] * occs[q] * occs[r] * (1/6) * me
                        f_buffer[p, s] += occs[q] * occs[r] * 0.5 * me
                gamma_buffer[p, q, s, t] += me
        
        return e0

    benchmark(run_legacy_loops)
