import pytest
import numpy as np
from numpy.testing import assert_allclose
from NuLattice.IMSRG import commutators as comm

@pytest.fixture
def imsrg_setup():
    """Provides a consistent model space for commutator testing."""
    dim = 8
    np.random.seed(42)
    return {
        "dim": dim,
        "occs": np.array([1.0]*4 + [0.0]*4), # 4 holes, 4 particles
        "a1": np.random.rand(dim, dim),
        "b1": np.random.rand(dim, dim),
        "a2": np.random.rand(dim, dim, dim, dim),
        "b2": np.random.rand(dim, dim, dim, dim)
    }

@pytest.mark.parametrize("func_base", ["antisymmetrize_2b_pq", "antisymmetrize_2b_rs"])
def test_antisymmetry_equivalence(imsrg_setup, func_base):
    """Verifies metadata-based swapaxes matches original contraction."""
    a2 = imsrg_setup["a2"]
    orig_func = getattr(comm, f"_{func_base}_original")
    np_func = getattr(comm, f"_{func_base}_np")
    
    assert_allclose(np_func(a2), orig_func(a2), atol=1e-15)

@pytest.mark.a2
def test_antisymmetry_einsum(benchmark, imsrg_setup):
    a2 = imsrg_setup["a2"]
    def f(x):
        return comm._antisymmetrize_2b_rs_original(comm._antisymmetrize_2b_pq_original(x))
    benchmark(f, a2)
@pytest.mark.a2
def test_antisymmetry_swapaxes(benchmark, imsrg_setup):
    a2 = imsrg_setup["a2"]
    def f(x):
        return comm._antisymmetrize_2b_rs_np(comm._antisymmetrize_2b_pq_np(x))
    benchmark(f, a2)

def test_comm_221_equivalence(imsrg_setup):
    """Verifies that the reshaped matmul version matches the loop-heavy original."""
    setup = imsrg_setup
    res_orig = comm._evaluate_comm_221_original(setup["occs"], setup["a2"], setup["b2"])
    res_np = comm._evaluate_comm_221_np(setup["occs"], setup["a2"], setup["b2"])
    
    assert_allclose(res_np, res_orig, atol=1e-15)

@pytest.mark.comm_221
def test_comm_221_einsum(benchmark, imsrg_setup):
    setup = imsrg_setup
    benchmark(comm._evaluate_comm_221_original, setup["occs"], setup["a2"], setup["b2"])

@pytest.mark.comm_221
def test_comm_221_np(benchmark, imsrg_setup):
    setup = imsrg_setup
    benchmark(comm._evaluate_comm_221_np, setup["occs"], setup["a2"], setup["b2"])

def test_comm_222_pphh_equivalence(imsrg_setup):
    """Checks the O(N^6) pphh bottleneck for numerical consistency."""
    setup = imsrg_setup
    res_orig = comm._evaluate_comm_222_pphh_original(setup["occs"], setup["a2"], setup["b2"])
    res_np = comm._evaluate_comm_222_pphh_np(setup["occs"], setup["a2"], setup["b2"])
    
    assert_allclose(res_np, res_orig, atol=1e-15)

@pytest.mark.comm_222_pphh
def test_comm_222_pphh_original(benchmark, imsrg_setup):
    setup = imsrg_setup
    benchmark(comm._evaluate_comm_222_pphh_original, setup["occs"], setup["a2"], setup["b2"])

@pytest.mark.comm_222_pphh
def test_comm_222_pphh_np(benchmark, imsrg_setup):
    setup = imsrg_setup
    benchmark(comm._evaluate_comm_222_pphh_np, setup["occs"], setup["a2"], setup["b2"])

def test_comm_222_ph_equivalence(imsrg_setup):
    """Checks the particle-hole sector after index transpositions."""
    setup = imsrg_setup
    res_orig = comm._evaluate_comm_222_ph_original(setup["occs"], setup["a2"], setup["b2"])
    res_np = comm._evaluate_comm_222_ph_np(setup["occs"], setup["a2"], setup["b2"])
    
    assert_allclose(res_np, res_orig, atol=1e-15)

@pytest.mark.comm_222
def test_comm_222_benchmark(benchmark, imsrg_setup):
    """Benchmarks the peak throughput of the reshaped matmul implementation."""
    setup = imsrg_setup
    benchmark(comm._evaluate_comm_222_np, setup["occs"], setup["a2"], setup["b2"])
