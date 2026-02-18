import pytest
import numpy as np
from numpy.testing import assert_allclose
from NuLattice.IMSRG import generator as gen

@pytest.fixture
def gen_setup():
    """Provides consistent model space for generator testing."""
    dim = 16
    np.random.seed(1337)
    
    # Mock Fock matrix (diagonal dominates for SP energies)
    f = np.diag(np.linspace(-20, 20, dim)) + np.random.rand(dim, dim) * 0.1
    
    # Mock Two-body interaction
    gamma = np.random.rand(dim, dim, dim, dim)
    
    return {
        "dim": dim,
        "occs": np.array([1.0]*8 + [0.0]*8), # 8 holes, 8 particles
        "f": f,
        "gamma": gamma,
        "delta": 0.05
    }

def test_build_1b_energy_difference_equivalence(gen_setup):
    """Verifies (N,1)-(1,N) broadcasting vs original einsum construction."""
    setup = gen_setup
    res_orig = gen._build_1b_energy_difference_original(setup["occs"], setup["f"], setup["delta"])
    res_np = gen._build_1b_energy_difference_np(setup["occs"], setup["f"], setup["delta"])
    
    assert_allclose(res_np, res_orig, atol=1e-15)

@pytest.mark.build_1b
def test_build_1b_energy_difference_equivalence_original(benchmark, gen_setup):
    """Benchmarks the single-pass N^4 construction."""
    setup = gen_setup
    benchmark(gen._build_1b_energy_difference_original, setup["occs"], setup["f"], setup["delta"])

@pytest.mark.build_1b
def test_build_1b_generator_difference_equivalence_np(benchmark, gen_setup):
    """Benchmarks the single-pass N^4 construction."""
    setup = gen_setup
    benchmark(gen._build_1b_energy_difference_np, setup["occs"], setup["f"], setup["delta"])

def test_build_2b_energy_difference_equivalence(gen_setup):
    """Verifies O(N^4) multi-dimensional broadcasting."""
    setup = gen_setup
    res_orig = gen._build_2b_energy_difference_original(setup["occs"], setup["f"], setup["delta"])
    res_np = gen._build_2b_energy_difference_np(setup["occs"], setup["f"], setup["delta"])
    
    assert_allclose(res_np, res_orig, atol=1e-15)

@pytest.mark.build_2b
def test_build_2b_energy_difference_equivalence_original(benchmark, gen_setup):
    """Benchmarks the single-pass N^4 construction."""
    setup = gen_setup
    benchmark(gen._build_2b_energy_difference_original, setup["occs"], setup["f"], setup["delta"])

@pytest.mark.build_2b
def test_build_2b_generator_difference_equivalence_np(benchmark, gen_setup):
    """Benchmarks the single-pass N^4 construction."""
    setup = gen_setup
    benchmark(gen._build_2b_energy_difference_np, setup["occs"], setup["f"], setup["delta"])

def test_build_2b_arctan_generator_equivalence(gen_setup):
    """Verifies boolean masking vs floating-point einsum masks."""
    setup = gen_setup
    res_orig = gen._build_2b_arctan_generator_original(
        setup["occs"], setup["f"], setup["gamma"], setup["delta"]
    )
    res_np = gen._build_2b_arctan_generator_np(
        setup["occs"], setup["f"], setup["gamma"], setup["delta"]
    )
    
    assert_allclose(res_np, res_orig, atol=1e-15)

@pytest.mark.generator
def test_2b_generator_original(benchmark, gen_setup):
    """Benchmarks the single-pass N^4 construction."""
    setup = gen_setup
    benchmark(
        gen._build_2b_arctan_generator_original, 
        setup["occs"], setup["f"], setup["gamma"], setup["delta"]
    )

@pytest.mark.generator
def test_2b_generator_np(benchmark, gen_setup):
    """Benchmarks the single-pass N^4 construction."""
    setup = gen_setup
    benchmark(
        gen._build_2b_arctan_generator_np, 
        setup["occs"], setup["f"], setup["gamma"], setup["delta"]
    )
