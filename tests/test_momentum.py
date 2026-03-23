import pytest
import numpy as np
from numpy.testing import assert_allclose
from NuLattice import lattice

@pytest.fixture
def p_setup():
    myL = 3
    return {
        "myL": myL,
        "sites": lattice.get_lattice(myL),
        "spin": 2,
        "isospin": 2
    }

@pytest.mark.parametrize("dim, func_name", [
    (0, "p_x"),
    (1, "p_y"),
    (2, "p_z")
])

def test_momentum_equivalence(p_setup, dim, func_name):
    """Checks that vectorized p_i matches the original loop implementation."""
    setup = p_setup
    
    new_func = getattr(lattice, func_name)
    orig_func = getattr(lattice, f"_{func_name}_original")

    res_orig = orig_func(setup["sites"], setup["myL"])
    res_new = new_func(setup["sites"], setup["myL"])
    
    res_orig.sort(key=lambda x: (x[0], x[1]))
    res_new.sort(key=lambda x: (x[0], x[1]))
    
    assert len(res_orig) == len(res_new)
    assert_allclose(res_new, res_orig, atol=1e-15)

def test_momentum_anti_hermiticity(p_setup):
    """
    The momentum operator p is Hermitian (p = p†).
    In this lattice representation: <i|p|j> = (<j|p|i>)*
    Since p_x = -0.5j for right hops and +0.5j for left hops,
    the matrix should be Hermitian.
    """
    setup = p_setup
    mat = lattice.p_x(setup["sites"], setup["myL"])
    
    lookup = {(int(r[0]), int(r[1])): r[2] for r in mat}
    
    # Hermitian condition: M_ij = conj(M_ji)
    for (i, j), val in lookup.items():
        assert (j, i) in lookup, f"Missing partner for ({i}, {j})"
        assert_allclose(val, np.conj(lookup[(j, i)]), atol=1e-15)

def test_momentum_directionality(p_setup):
    """Verifies that p_x only changes the x-coordinate of the state."""
    setup = p_setup
    basis = lattice.get_sp_basis(setup["myL"])
    
    mat = lattice.p_y(setup["sites"], setup["myL"])
    
    for i, j, val in mat:
        state_i = basis[int(i)]
        state_j = basis[int(j)]
        
        assert state_i[0] == state_j[0], "p_y changed the x coordinate!"
        assert state_i[2] == state_j[2], "p_y changed the z coordinate!"
        assert state_i[3] == state_j[3], "p_y changed isospin!"
        assert state_i[4] == state_j[4], "p_y changed spin!"
        
        diff = abs(state_i[1] - state_j[1])
        assert diff in [1, setup["myL"] - 1], "p_y hop distance is not 1!"

@pytest.mark.px
def test_px(benchmark):
    """Benchmarks the vectorized p_x implementation."""
    myL = 16
    sites = lattice.get_lattice(myL)
    benchmark(lattice._p_x_original, sites, myL)

@pytest.mark.px
def test_px_np(benchmark):
    """Benchmarks the vectorized p_x implementation."""
    myL = 16
    sites = lattice.get_lattice(myL)
    benchmark(lattice.p_x, sites, myL)
