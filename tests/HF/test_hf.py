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
