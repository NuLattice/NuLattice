import pytest
from NuLattice import lattice

@pytest.mark.lattice_generation
def test_get_lattice_list(benchmark):
    """Benchmarks the original loop-based lattice generation."""
    L = 16 
    res = benchmark(lattice._get_lattice_list, L)
    assert len(res) == L**3

@pytest.mark.lattice_generation
def test_get_lattice_mgrid(benchmark):
    """Benchmarks the optimized mgrid-based lattice generation."""
    L = 16
    res = benchmark(lattice._get_lattice_mgrid, L)
    assert res.shape == (L**3, 3)

@pytest.mark.benchmark(group="basis_generation")
def test_get_sp_basis_list(benchmark):
    """Benchmarks the original nested loops for SP basis."""
    L = 8
    res = benchmark(lattice._get_sp_basis_list, L)
    assert len(res) == L**3 * 4

@pytest.mark.benchmark(group="basis_generation")
def test_get_sp_basis_mgrid(benchmark):
    """Benchmarks the optimized vectorized SP basis."""
    L = 8
    res = benchmark(lattice._get_sp_basis_mgrid, L)
    assert len(res) == L**3 * 4

@pytest.mark.benchmark(group="state2index")
def test_state2index_naive_speed(benchmark):
    myL = 16
    basis = lattice.get_sp_basis(myL)
    benchmark(lambda: [lattice._state2index_original(state, myL) for state in basis])


@pytest.mark.benchmark(group="state2index")
def test_state2index_stride_speed(benchmark):
    myL = 16
    basis = lattice.get_sp_basis(myL)
    benchmark(lambda: [lattice._state2index_strided(state, myL) for state in basis])

@pytest.mark.boundary_logic
def test_right_if(benchmark):
    L = 16
    benchmark(lambda: [lattice._right_if(i, L) for i in range(L)])

@pytest.mark.boundary_logic
def test_right_modulo(benchmark):
    L = 16
    benchmark(lambda: [lattice._right_modulus(i, L) for i in range(L)])

@pytest.mark.contacts
def test_contacts_original(benchmark):
    """
    Benchmarks the speedup of the vectorized contacts.
    """
    p = {
        "vT1": 1.5,
        "vS1": -2.0,
        "myL": 2,
        "spin": 2,
        "isospin": 2
    }

    p["myL"] = 4
    lat = lattice.get_lattice(p["myL"])
    benchmark(lattice._contacts_original, p["vT1"], p["vS1"], lat, p["myL"])

@pytest.mark.contacts
def test_contacts_np(benchmark):
    """
    Benchmarks the speedup of the vectorized contacts.
    """
    p = {
        "vT1": 1.5,
        "vS1": -2.0,
        "myL": 2,
        "spin": 2,
        "isospin": 2
    }

    p["myL"] = 4
    lat = lattice.get_lattice(p["myL"])
    benchmark(lattice._contacts_np, p["vT1"], p["vS1"], lat, p["myL"])
