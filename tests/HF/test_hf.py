import pytest
import numpy as np
from numpy.testing import assert_allclose
from NuLattice.HF import hartree_fock as hf

@pytest.fixture
def hf_setup():
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
def test_contract_3nf_equivalence(hf_setup):
    """
    Verifies that the Numba/Vectorized version matches 
    the original 36-hardcoded-operation version.
    """
    setup = hf_setup
    res_orig = hf._contract_3nf_original(setup["w3"], setup["dens"])
    res_opt = hf._contract_3nf_np(setup["w3"], setup["dens"])
    assert_allclose(res_opt, res_orig, atol=1e-15, err_msg="np 3NF contraction does not match original!")

@pytest.mark.contract_3nf
def test_contract_3nf_benchmark_original(benchmark, hf_setup):
    benchmark(hf._contract_3nf_original, hf_setup["w3"], hf_setup["dens"])

@pytest.mark.contract_3nf
def test_contract_3nf_benchmark_np(benchmark, hf_setup):
    benchmark(hf._contract_3nf_np, hf_setup["w3"], hf_setup["dens"])
