import pytest
import NuLattice.lattice_soa as lat
import NuLattice.references as ref

@pytest.fixture
def hf_setup():
    """Pre-calculates operators so the benchmark focuses on the HF solver logic."""
    thisL = 6
    a_lat = 2.5
    phys_unit = lat.phys_unit(a_lat)
    
    my_basis = lat.get_sp_basis(thisL)
    n_states = len(my_basis)
    lattice = lat.get_lattice(thisL)
    
    # Generate SoA Operators
    # Using the internal _np functions to get SoA objects directly
    myTkin = lat._Tkin(lattice, thisL)
    mycontact = lat._contacts(-9.0, -9.0, lattice, thisL)
    my3body = lat._NNNcontact(6.0, lattice, thisL)
    
    # Setup 16O ground state
    my_ref = ref.ref_16O_gs
    hole = ref.reference_to_holes(my_ref, my_basis)
    # dens = hf.init_density(nstat, hole)
    
    return {
        "ops": (myTkin, mycontact, my3body),
        "n_states": n_states,
        "hole": hole,
        # "dens": dens,
        "phys_unit": phys_unit
    }

@pytest.mark.benchmark(group="HF_Flow")
def test_hf(benchmark, hf_setup):
    import NuLattice.HF.hartree_fock as hf
    """Benchmarks the solve_HF function performance."""
    ops = hf_setup["ops"]
    n_states, hole = hf_setup["n_states"], hf_setup["hole"]
    dens = hf.init_density(n_states, hole)
    def run_solver():
        erg, trafo, conv = hf.solve_HF(
            *(op.to_list() for op in ops), dens, mix=0.7, eps=1.e-8, max_iter=100, verbose=False
        )
        return erg

    result_erg = benchmark(run_solver)
    assert result_erg is not None

@pytest.mark.benchmark(group="HF_Flow")
def test_hf_soa(benchmark, hf_setup):
    import NuLattice.HF.hfsoa as hf
    """Benchmarks the solve_HF function performance."""
    ops = hf_setup["ops"]
    n_states, hole = hf_setup["n_states"], hf_setup["hole"]
    dens = hf.init_density(n_states, hole)
    def run_solver():
        erg, trafo, conv = hf.solve_HF(
            *ops, dens, mix=0.7, eps=1.e-8, max_iter=100, verbose=False
        )
        return erg

    result_erg = benchmark(run_solver)
    assert result_erg is not None
