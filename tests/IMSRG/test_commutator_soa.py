import pytest
import numpy as np
import NuLattice.IMSRG.commutators as comm
import NuLattice.IMSRG.soa.commutators as comm_soa

@pytest.fixture
def comm_setup():
    """Sets up realistic IMSRG(2) tensors for a small lattice (L=3, N=216)."""
    # We use a smaller N here because O(N^6) grows extremely fast.
    # N=216 means N^6 is ~100 billion operations.
    L = 3 
    dim = 2 * (L**3) 
    
    np.random.seed(42)
    occs = np.zeros(dim)
    occs[:8] = 1.0  # 8 occupied states (e.g., Oxygen-like core)
    
    a1 = np.random.rand(dim, dim)
    b1 = np.random.rand(dim, dim)
    
    # 2-body tensors (N, N, N, N)
    # We use float32 for L=3 to keep memory footprint sane during testing
    a2 = np.random.rand(dim, dim, dim, dim).astype(np.float64)
    b2 = np.random.rand(dim, dim, dim, dim).astype(np.float64)
    
    return occs, a1, a2, b1, b2

@pytest.mark.benchmark(group="Commutators_O6")
def test_comm_222_np(benchmark, comm_setup):
    """Benchmarks the Reshape + NumPy GEMM implementation."""
    occs, a1, a2, b1, b2 = comm_setup
    
    def run_np():
        # Using your previous _np version which uses np.matmul on reshaped arrays
        res = comm._evaluate_comm_222_pphh_np(occs, a2, b2)
        return res

    benchmark(run_np)

@pytest.mark.benchmark(group="Commutators_O6")
def test_comm_222_kernel(benchmark, comm_setup):
    """Benchmarks the Numba Parallel Kernel implementation."""
    occs, a1, a2, b1, b2 = comm_setup
    
    # Force a warm-up compilation before benchmarking
    _ = comm_soa._evaluate_comm_222_pphh(occs, a2, b2)

    def run_kernel():
        # Using the new Numba prange version
        res = comm_soa._evaluate_comm_222_pphh(occs, a2, b2)
        return res

    benchmark(run_kernel)

@pytest.mark.benchmark(group="Commutators_Full")
def test_full_imsrg_comm_np(benchmark, comm_setup):
    """Benchmarks the full IMSRG(2) commutator using NumPy logic."""
    occs, a1, a2, b1, b2 = comm_setup
    
    def run_full_np():
        return comm._evaluate_imsrg_commutator_np(occs, a1, a2, b1, b2)

    benchmark(run_full_np)

@pytest.mark.benchmark(group="Commutators_Full")
def test_full_imsrg_comm_kernel(benchmark, comm_setup):
    """Benchmarks the full IMSRG(2) commutator using Numba logic."""
    occs, a1, a2, b1, b2 = comm_setup
    
    # Warm up
    _ = comm_soa._evaluate_imsrg2_commutator(occs, a1, a2, b1, b2)

    def run_full_kernel():
        return comm_soa._evaluate_imsrg2_commutator(occs, a1, a2, b1, b2)

    benchmark(run_full_kernel)
