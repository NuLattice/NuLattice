import pytest
from NuLattice.IMSRG import ode_solver
from NuLattice.IMSRG.soa import torch_solver

EXPECTED_MEV = -19.916
TOLERANCE_MEV = 1e-3

def test_imsrg_accuracy(helium3_hamiltonian):
    """Verify the Torch solver hits the correct physical target."""
    h = helium3_hamiltonian
    
    e_lattice, _ = torch_solver.solve_imsrg2(
        h["occs"], h["e0"], h["f"], h["gamma"], 
        s_max=40.0, eta_criterion=1e-3, track_data=False
    )
    
    e_mev = e_lattice * h["phys_unit"]
    assert e_mev == pytest.approx(EXPECTED_MEV, abs=TOLERANCE_MEV)

@pytest.mark.benchmark(group="imsrg-solvers")
def test_benchmark_torch_solver(benchmark, helium3_hamiltonian):
    """Benchmark the M4-optimized Torch solver."""
    h = helium3_hamiltonian
    
    # Warm up (to handle torch.compile overhead)
    torch_solver.solve_imsrg2(h["occs"], h["e0"], h["f"], h["gamma"], s_max=1.0, track_data=False)
    
    def run_torch():
        return torch_solver.solve_imsrg2(
            h["occs"], h["e0"], h["f"], h["gamma"], 
            s_max=10.0, eta_criterion=1e-3, track_data=False
        )
    
    result_e, _ = benchmark(run_torch)
    assert result_e < 0

@pytest.mark.benchmark(group="imsrg-solvers")
def test_benchmark_original_solver(benchmark, helium3_hamiltonian):
    """Benchmark the original NumPy-based solver for comparison."""
    h = helium3_hamiltonian
    
    occs_np = h["occs"].numpy()
    f_np = h["f"].numpy()
    g_np = h["gamma"].numpy()
    
    def run_orig():
        return ode_solver.solve_imsrg2(
            occs_np, h["e0"], f_np, g_np, 
            s_max=10.0, eta_criterion=1e-3
        )
    
    benchmark(run_orig)
