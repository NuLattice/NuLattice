import pytest
import numpy as np
import NuLattice.lattice_soa as lat
import NuLattice.references as ref

def test_imsrg_result_equivalence():
    """
    Verification test to ensure SoA + NJIT logic produces the exact same 
    energy results as the legacy AoS implementation.
    """
    # 1. Setup shared physical parameters
    L = 2
    my_basis = lat.get_sp_basis(L)
    lattice = lat.get_lattice(L)
    
    # 2. Generate Operators in both formats
    # SoA version (New)
    op1_soa = lat._Tkin(lattice, L)
    op2_soa = lat._contacts(-9.0, -9.0, lattice, L)
    op3_soa = lat._NNNcontact(6.0, lattice, L)
    
    # AoS version (Legacy)
    op1_aos = op1_soa.to_list()
    op2_aos = op2_soa.to_list()
    op3_aos = op3_soa.to_list()
    
    # 3. Setup Occupations
    he3_ref = ref.ref_3He_gs
    from NuLattice.IMSRG.normal_ordering import create_occupations
    occs = create_occupations(my_basis, he3_ref)
    
    # 4. Execute Legacy (AoS) Flow
    from NuLattice.IMSRG import normal_ordering as no_aos
    from NuLattice.IMSRG import ode_solver
    
    e0_aos, f_aos, gamma_aos = no_aos.compute_normal_ordered_hamiltonian_no2b(
        occs, op1_aos, op2_aos, op3_aos
    )
    e_imsrg_aos, _ = ode_solver.solve_imsrg2(occs, e0_aos, f_aos, gamma_aos, s_max=10)

    # 5. Execute Optimized (SoA) Flow
    import NuLattice.IMSRG.soa.normal_ordering as no_soa
    
    e0_soa, f_soa, gamma_soa = no_soa.compute_normal_ordered_hamiltonian_soa(
        occs, op1_soa, op2_soa, op3_soa
    )
    e_imsrg_soa, _ = ode_solver.solve_imsrg2(occs, e0_soa, f_soa, gamma_soa, s_max=10)

    # 6. Assertions (Checking for numerical equivalence)
    # Check Normal Ordering Energy
    assert e0_soa == pytest.approx(e0_aos, rel=1e-12), "Initial E0 mismatch"
    
    # Check Fock Matrix (1-body)
    np.testing.assert_allclose(f_soa, f_aos, atol=1e-12, err_msg="Fock matrix mismatch")
    
    # Check Effective Interaction (2-body)
    np.testing.assert_allclose(gamma_soa, gamma_aos, atol=1e-12, err_msg="Gamma tensor mismatch")
    
    # Check Final Converged Energy
    assert e_imsrg_soa == pytest.approx(e_imsrg_aos, rel=1e-10), "Final IMSRG Energy mismatch"
