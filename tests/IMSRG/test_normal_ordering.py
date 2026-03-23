import pytest
import numpy as np
from numpy.testing import assert_allclose
from NuLattice.IMSRG import normal_ordering as no

@pytest.fixture
def no_setup():
    """Provides consistent basis and interaction lists for normal ordering tests."""
    dim = 8
    np.random.seed(42)
    
    # Mock occupation numbers (4 holes, 4 particles)
    occs = np.array([1.0]*4 + [0.0]*4)
    
    # Mock h1: [p, q, val]
    h1 = [[i, i, -2.0 * i] for i in range(dim)]
    
    # Mock h2: [p, q, r, s, val] (non-antisymmetrized input)
    h2 = []
    for _ in range(20):
        p, q, r, s = np.random.randint(0, dim, 4)
        if p != q and r != s:
            h2.append((p, q, r, s, np.random.rand()))

    # p, q, r, s, t, u, v
    h3 = [(0, 1, 2, 3, 4, 5, 0.5)]   
    return {
        "dim": dim,
        "occs": occs,
        "h1": h1,
        "h2": h2,
        "h3": h3
    }

@pytest.mark.benchmark(group="expand_h2")
def test_expand_h2_equivalence(no_setup):
    """Verifies vectorized tensor population matches the manual list expansion."""
    setup = no_setup
    
    res_orig_list = no._expand_h2_original(setup["h2"])
    res_orig_tensor = np.zeros((setup["dim"],) * 4)
    for p, q, r, s, me in res_orig_list:
        res_orig_tensor[p, q, r, s] += me
        
    res_np_tensor = no._expand_h2_np(setup["h2"])
    
    assert_allclose(res_np_tensor, res_orig_tensor, atol=1e-15)

@pytest.mark.benchmark(group="expand_h2")
def test_expand_h2_original(benchmark, no_setup):
    setup = no_setup
    benchmark(no._expand_h2_original, setup["h2"])

@pytest.mark.benchmark(group="expand_h2")
def test_expand_h2_np(benchmark, no_setup):
    setup = no_setup
    benchmark(no._expand_h2_np, setup["h2"])

@pytest.mark.benchmark(group="expand_h3")
def test_expand_h3_equivalence(no_setup):
    """Verifies vectorized tensor population matches the manual list expansion."""
    setup = no_setup
    
    res_orig_list = no._expand_h3_original(setup["h3"])
    dim = 6
    res_orig_tensor = np.zeros((dim, ) * 6)
    for p, q, r, s, t, u, me in res_orig_list:
        res_orig_tensor[p, q, r, s, t, u] += me
        
    res_np_tensor = no._expand_h3_np(setup["h3"])
    
    assert_allclose(res_np_tensor, res_orig_tensor, atol=1e-15)

@pytest.mark.benchmark(group="expand_h3")
def test_expand_h3_original(benchmark, no_setup):
    setup = no_setup
    benchmark(no._expand_h3_original, setup["h3"])

@pytest.mark.benchmark(group="expand_h3")
def test_expand_h3_np(benchmark, no_setup):
    setup = no_setup
    benchmark(no._expand_h3_np, setup["h3"])


def test_normal_ordered_hamiltonian_equivalence(no_setup):
    """Checks that e0 and f match after replacing Python loops with einsum."""
    setup = no_setup
    
    e0_orig, f_orig, gamma_orig = no._compute_normal_ordered_hamiltonian_no2b_original(
        setup["occs"], setup["h1"], setup["h2"]
    )
    
    e0_np, f_np, gamma_np = no._compute_normal_ordered_hamiltonian_np(
        setup["occs"], setup["h1"], setup["h2"]
    )
    
    assert_allclose(e0_np, e0_orig, atol=1e-15)
    assert_allclose(f_np, f_orig, atol=1e-15)
    assert_allclose(gamma_np, gamma_orig, atol=1e-15)

@pytest.mark.benchmark(group="normal_ordering")
def test_normal_ordering_benchmark(benchmark, no_setup):
    """Benchmarks the transition from list-processing to tensor-contractions."""
    setup = no_setup
    benchmark(
        no._compute_normal_ordered_hamiltonian_np, 
        setup["occs"], setup["h1"], setup["h2"]
    )
