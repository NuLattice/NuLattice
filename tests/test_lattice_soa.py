import pytest
import numpy as np
from NuLattice import lattice as aos
from NuLattice import lattice_soa as soa

def canonicalize_operator_list(op_list):
    return sorted(op_list, key=lambda x: tuple(x[:-1]))

def compare_lists_approx(list_a, list_b, tol=1e-10):
    """
    Compares two lists of tuples/lists [i, j, ..., val] for approximate equality.
    """
    sorted_a = canonicalize_operator_list(list_a)
    sorted_b = canonicalize_operator_list(list_b)

    assert len(sorted_a) == len(sorted_b), f"Length mismatch: {len(sorted_a)} vs {len(sorted_b)}"

    for row_a, row_b in zip(sorted_a, sorted_b):
        assert row_a[:-1] == row_b[:-1], f"Index mismatch: {row_a} vs {row_b}"
        np.testing.assert_allclose(row_a[-1], row_b[-1], atol=tol, err_msg=f"Value mismatch at {row_a[:-1]}")

@pytest.fixture
def lattice_params():
    return {
        "myL": 3,
        "spin": 2,
        "isospin": 2
    }

class TestBasisEquivalence:
    def test_sp_basis(self, lattice_params):
        """Test that Single Particle Basis generation is identical."""
        basis_aos = aos.get_sp_basis(**lattice_params)
        basis_soa = soa.get_sp_basis(**lattice_params)
        if isinstance(basis_soa, np.ndarray):
            basis_soa = basis_soa.tolist()
            
        assert len(basis_aos) == len(basis_soa)
        
        basis_aos.sort()
        basis_soa.sort()
        
        assert basis_aos == basis_soa

class TestOneBodyEquivalence:
    def test_kinetic_energy(self, lattice_params):
        """Test Tkin (Kinetic Energy) equivalence."""
        lattice_sites_aos = aos.get_lattice(lattice_params["myL"])
        
        t_aos = aos.Tkin(lattice_sites_aos, **lattice_params)
        
        t_soa = soa.Tkin(lattice_sites_aos, **lattice_params)
        
        # 3. Compare
        compare_lists_approx(t_aos, t_soa.to_list())

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_momentum_operators(self, lattice_params, axis):
        """Test p_x, p_y, p_z equivalence."""
        lattice_sites = aos.get_lattice(lattice_params["myL"])
        
        p_soa = getattr(soa, f"p_{axis}")(lattice_sites, **lattice_params)
        p_aos = getattr(aos, f"p_{axis}")(lattice_sites, **lattice_params)
        
        compare_lists_approx(p_aos, p_soa)

class TestTwoBodyEquivalence:
    def test_contacts(self, lattice_params):
        """Test 2-Body Contact Interactions."""
        lattice_sites = aos.get_lattice(lattice_params["myL"])
        vT1 = -2.5
        vS1 = -1.5
        
        v_aos = aos.contacts(vT1, vS1, lattice_sites, **lattice_params)
        
        v_soa_op = soa._contacts(vT1, vS1, lattice_sites, **lattice_params)
        v_soa = v_soa_op.to_list()
        
        compare_lists_approx(v_aos, v_soa)

class TestThreeBodyEquivalence:
    def test_nnn_contact(self, lattice_params):
        """Test 3-Body (NNN) Contact Interactions."""
        lattice_sites = aos.get_lattice(lattice_params["myL"])
        v3NF = 5.0
        
        w_aos = aos.NNNcontact(v3NF, lattice_sites, **lattice_params)
        
        w_soa_op = soa._NNNcontact(v3NF, lattice_sites, **lattice_params)
        w_soa = w_soa_op.to_list()
        
        compare_lists_approx(w_aos, w_soa)

class TestOperatorClassFeatures:
    """Tests specific to the SoA class features beyond simple list equivalence."""
    
    def test_one_body_to_dense(self, lattice_params):
        """Ensure OneBodyOperator.to_dense matches numpy construction."""
        lattice_sites = aos.get_lattice(lattice_params["myL"])
        
        t_soa_op = soa._Tkin(lattice_sites, **lattice_params)
        
        dense_mat = t_soa_op.to_dense()
        
        assert dense_mat.shape == (t_soa_op.nstat, t_soa_op.nstat)
        
        np.testing.assert_allclose(np.diag(dense_mat), 6.0)
        np.testing.assert_allclose(dense_mat, dense_mat.T)

    def test_soa_type_safety(self, lattice_params):
        """Ensure indices are strictly integers in the internal storage."""
        lattice_sites = aos.get_lattice(lattice_params["myL"])
        t_soa_op = soa._Tkin(lattice_sites, **lattice_params)
        
        assert t_soa_op.indices.dtype == np.int64
