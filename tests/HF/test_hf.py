import pytest
import numpy as np
from numpy.testing import assert_allclose
from NuLattice.HF import hartree_fock as hf


@pytest.fixture
def hf_1nf_2nf_setup():
    """Setup for 1-body and 2-body contraction testing."""
    nstat = 6
    np.random.seed(42)
    
    # 1-body elements: [p, q, val]
    t_kin = []
    for i in range(nstat):
        for j in range(nstat):
            if np.random.rand() > 0.5:
                t_kin.append([i, j, np.random.normal()])
                
    # 2-body elements: [p, q, r, s, val]
    v2 = []
    for _ in range(30):
        indices = np.random.randint(0, nstat, 4).tolist()
        val = np.random.normal()
        v2.append(indices + [val])
        
    # Density matrix
    tmp = np.random.rand(nstat, nstat)
    dens = (tmp + tmp.T) / 2.0
    
    return {
        "nstat": nstat,
        "t_kin": t_kin,
        "v2": v2,
        "dens": dens
    }

@pytest.fixture
def hf_3nf_setup():
    nstat = 10 
    np.random.seed(42)
    
    # Mock density matrix (symmetric, trace = A)
    # random symmetric matrix to ensure all sectors of contract_3nf are touched
    tmp = np.random.rand(nstat, nstat)
    dens = (tmp + tmp.T) / 2.0
    
    # Mock w3: [a, b, c, d, e, f, val]
    # generate a few elements where indices might overlap to check summation
    w3 = []
    for _ in range(50):
        indices = np.random.randint(0, nstat, 6).tolist()
        val = np.random.normal(0, 1.0)
        w3.append(indices + [val])
        
    return {
        "nstat": nstat,
        "dens": dens,
        "w3": w3
    }

@pytest.mark.contract_3nf
def test_contract_3nf_equivalence(hf_3nf_setup):
    """
    Verifies that the Numba/Vectorized version matches 
    the original 36-hardcoded-operation version.
    """
    setup = hf_3nf_setup
    res_orig = hf._contract_3nf_original(setup["w3"], setup["dens"])
    res_opt = hf._contract_3nf_np(setup["w3"], setup["dens"])
    assert_allclose(res_opt, res_orig, atol=1e-15, err_msg="np 3NF contraction does not match original!")

@pytest.mark.contract_3nf
def test_contract_3nf_benchmark_original(benchmark, hf_3nf_setup):
    benchmark(hf._contract_3nf_original, hf_3nf_setup["w3"], hf_3nf_setup["dens"])

@pytest.mark.contract_3nf
def test_contract_3nf_benchmark_np(benchmark, hf_3nf_setup):
    benchmark(hf._contract_3nf_np, hf_3nf_setup["w3"], hf_3nf_setup["dens"])


@pytest.mark.get_1body_matrix
def test_get_1body_matrix_equivalence(hf_1nf_2nf_setup):
    """Verifies vectorized matrix population vs manual loops."""
    setup = hf_1nf_2nf_setup
    
    res_orig = hf._get_1body_matrix_original(setup["t_kin"], setup["nstat"])
    res_np = hf._get_1body_matrix_np(setup["t_kin"], setup["nstat"])
    
    assert_allclose(res_np, res_orig, atol=1e-15)

@pytest.mark.get_1body_matrix
def test_get_1body_matrix_benchmark_original(benchmark, hf_1nf_2nf_setup):
    benchmark(hf._get_1body_matrix_original, hf_1nf_2nf_setup["t_kin"], hf_1nf_2nf_setup["nstat"])

@pytest.mark.get_1body_matrix
def test_get_1body_matrix_benchmark_np(benchmark, hf_1nf_2nf_setup):
    benchmark(hf._get_1body_matrix_np, hf_1nf_2nf_setup["t_kin"], hf_1nf_2nf_setup["nstat"])

@pytest.mark.contract_2nf
def test_contract_2nf_equivalence(hf_1nf_2nf_setup):
    """Verifies the four-term antisymmetry logic in 2NF contraction."""
    setup = hf_1nf_2nf_setup
    
    res_orig = hf._contract_2nf_original(setup["v2"], setup["dens"])
    res_opt = hf._contract_2nf_fastest(setup["v2"], setup["dens"])
    
    assert_allclose(res_opt, res_orig, atol=1e-15, 
                    err_msg="2NF contraction mismatch! Check antisymmetry sign conventions.")

@pytest.mark.contract_2nf
def test_contract_2nf_benchmark_original(benchmark, hf_1nf_2nf_setup):
    benchmark(hf._contract_2nf_original, hf_1nf_2nf_setup["v2"], hf_1nf_2nf_setup["dens"])

@pytest.mark.contract_2nf
def test_contract_2nf_benchmark_np(benchmark, hf_1nf_2nf_setup):
    benchmark(hf._contract_2nf_fastest, hf_1nf_2nf_setup["v2"], hf_1nf_2nf_setup["dens"])


@pytest.mark.init_density
def test_init_density(benchmark):
    nstat = 10
    hole = (0, 2, 5)
    benchmark(hf.init_density, nstat, hole)

# @pytest.mark.init_density
# def test_init_density_np(benchmark):
#     nstat = 10
#     hole = (0, 2, 5)
#     benchmark(hf.init_density_np, nstat, hole)

@pytest.mark.init_density
def test_init_density_diagonal_occupation():
    """Verifies that init_density creates the correct diagonal occupation matrix."""
    nstat = 10
    hole = (0, 2, 5)
    
    dens = hf.init_density(nstat, hole)
    
    assert dens.shape == (nstat, nstat)
    
    assert np.trace(dens) == len(hole)
    
    for i in range(nstat):
        if i in hole:
            assert dens[i, i] == 1.0, f"State {i} should be occupied."
        else:
            assert dens[i, i] == 0.0, f"State {i} should be empty."
            
    off_diag = dens - np.diag(np.diag(dens))
    assert np.all(off_diag == 0.0), "Density matrix must be diagonal at initialization."


@pytest.mark.init_density
def test_init_density_empty():
    """Ensures the function handles zero occupied states (vacuum)."""
    nstat = 4
    hole = ()
    dens = hf.init_density(nstat, hole)
    
    assert_allclose(dens, np.zeros((4, 4)))

@pytest.mark.init_density
def test_init_density_full():
    """Ensures it handles a completely filled basis."""
    nstat = 3
    hole = (0, 1, 2)
    dens = hf.init_density(nstat, hole)
    
    assert_allclose(dens, np.eye(3))

@pytest.fixture
def hf_data():
    """Generates a consistent physical system for testing."""
    nstat = 40
    npart = 8
    np.random.seed(42)
    
    # 1-body: [p, q, val]
    op1 = [[i, i, float(i*0.5)] for i in range(nstat)]
    
    # 2-body: [p, q, r, s, val]
    op2 = []
    for _ in range(100):
        indices = np.random.randint(0, nstat, 4).tolist()
        op2.append(indices + [np.random.normal()])
        
    # 3-body: [p, q, r, s, t, u, val]
    op3 = []
    for _ in range(50):
        indices = np.random.randint(0, nstat, 6).tolist()
        op3.append(indices + [np.random.normal()])
        
    # Initial density
    dens = np.zeros((nstat, nstat))
    for i in range(npart):
        dens[i, i] = 1.0
        
    return {
        "op1": op1, "op2": op2, "op3": op3, 
        "dens": dens, "npart": npart
    }

@pytest.mark.hf_energy
def test_hf_energy_equivalence(hf_data):
    """Verifies that the optimized _np version yields identical physics to _original."""
    d = hf_data
    erg_orig = hf.HF_energy(d["op1"], d["op2"], d["op3"], d["dens"])
    erg_np = hf.HF_energy_np(d["op1"], d["op2"], d["op3"], d["dens"])
    assert pytest.approx(erg_np) == erg_orig
   

@pytest.mark.hf_energy
def test_hf_energy_speed_original(benchmark, hf_data):
    """Benchmarks the unoptimized Hartree-Fock iteration."""
    d = hf_data
    result = benchmark(hf.HF_energy, d["op1"], d["op2"], d["op3"], d["dens"])

# @pytest.mark.hf_energy
# def test_hf_energy_speed_np(benchmark, hf_data):
#     """Benchmarks the optimized NumPy/BLAS Hartree-Fock iteration."""
#     d = hf_data
#     result = benchmark(hf.HF_energy_np, d["op1"], d["op2"], d["op3"], d["dens"])
    
    assert result[0] is not None
@pytest.mark.hf_iter
def test_hf_iter_equivalence(hf_data):
    """Verifies that the optimized _np version yields identical physics to _original."""
    d = hf_data
    
    # Run original iteration
    e_orig, dens_orig, vecs_orig = hf._HF_iter_original(d["op1"], d["op2"], d["op3"], d["dens"])
    
    # Run optimized iteration
    e_np, dens_np, vecs_np = hf._HF_iter_np(d["op1"], d["op2"], d["op3"], d["dens"])
    
    # Check Energy (Scalar)
    assert pytest.approx(e_np) == e_orig
    
    np.testing.assert_allclose(dens_np, dens_orig, atol=1e-13, 
                               err_msg="Density matrices diverged between original and NP versions.")

@pytest.mark.hf_iter
def test_hf_iter_speed_original(benchmark, hf_data):
    """Benchmarks the unoptimized Hartree-Fock iteration."""
    d = hf_data
    result = benchmark(hf._HF_iter_original, d["op1"], d["op2"], d["op3"], d["dens"])
    assert result[0] is not None

@pytest.mark.hf_iter
def test_hf_iter_speed_np(benchmark, hf_data):
    """Benchmarks the optimized NumPy/BLAS Hartree-Fock iteration."""
    d = hf_data
    result = benchmark(hf._HF_iter_np, d["op1"], d["op2"], d["op3"], d["dens"])
    
    assert result[0] is not None

# @pytest.mark.solve_hf
# def test_solve_hf_equivalence(hf_data):
#     """Verifies that the optimized _np version yields identical physics to _original."""
#     d = hf_data
    
#     # Run original iteration
#     e_orig, dens_orig, vecs_orig = hf.solve_HF(d["op1"], d["op2"], d["op3"], d["dens"])
    
#     # Run optimized iteration
#     e_np, dens_np, vecs_np = hf.solve_HF_np(d["op1"], d["op2"], d["op3"], d["dens"])
    
#     # Check Energy (Scalar)
#     assert pytest.approx(e_np) == e_orig
    
#     np.testing.assert_allclose(dens_np, dens_orig, atol=1e-13, 
#                                err_msg="Density matrices diverged between original and NP versions.")

@pytest.mark.solve_hf
def test_solve_hf_speed(benchmark, hf_data):
    d = hf_data
    result = benchmark(hf.solve_HF, d["op1"], d["op2"], d["op3"], d["dens"])


# @pytest.mark.solve_hf
# def test_solve_hf_speed_np(benchmark, hf_data):
#     d = hf_data
#     result = benchmark(hf.solve_HF_np, d["op1"], d["op2"], d["op3"], d["dens"])

