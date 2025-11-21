import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent / ".." / ".."))
import NuLattice.lattice as lat
import NuLattice.references as ref
import NuLattice.CCM.coupled_cluster as ccm
import NuLattice.FCI.few_body_diagonalization as fbd
from NuLattice.IMSRG import normal_ordering
from NuLattice.IMSRG import ode_solver
from scipy.sparse.linalg import eigsh as arpack_eigsh
import numpy as np

myL = 3
params = [[2.5, -9, 6], [2.0, -8, 5.5], [1.7, -7, 4.4]]
lattice = lat.get_lattice(myL)
my_basis = lat.get_sp_basis(myL)
nstat =  len(my_basis)

# Flag to set whether we evaluate IMSRG(2)
# IMSRG(2) is the most expensive method to compute
with_imsrg2 = False

if with_imsrg2:
    print("Warning: Evaluating IMSRG(2) results for L=3. This will take 16 GB of memory and 10s of hours.")
else:
    print("Skipping IMSRG(2) because it is expensive. Set \"with_imsrg2 = True\" to also evaluate IMSRG(2) results.")

he3_fci_ens = []
he4_fci_ens = []
he3_ccsd_ens = []
he4_ccsd_ens = []
he3_imsrg_ens = []
he4_imsrg_ens = []

for i in params:
    a_lat, vT1, w3 = i
    vS1 = vT1
    phys_unit = lat.phys_unit(a_lat)

    myTkin=lat.Tkin(lattice, myL)
    mycontact=lat.contacts(vT1, vS1, lattice, myL)
    my3body=lat.NNNcontact(w3, lattice, myL)

    #FCI
    # Compute He3
    numpart=3
    tz = -1
    sz = -1
    He3_lookup = fbd.get_many_body_states(my_basis, numpart, total_tz=tz, total_sz=sz)
    T3_csr_mat = fbd.get_csr_matrix_scalar_op(He3_lookup, myTkin, nstat)
    V3_csr_mat = fbd.get_csr_matrix_scalar_op(He3_lookup, mycontact, nstat)
    W3_csr_mat = fbd.get_csr_matrix_scalar_op(He3_lookup, my3body, nstat)

    H3_csr_mat = T3_csr_mat + V3_csr_mat + W3_csr_mat

    k_eig=1
    vals, vecs = arpack_eigsh(H3_csr_mat, k=k_eig, which='SA')
    he3_fci_ens.append(np.round(vals[0] * phys_unit, 2))

    # Compute He4
    numpart=4
    tz = 0
    sz = 0
    He4_lookup = fbd.get_many_body_states(my_basis, numpart, total_tz=tz, total_sz=sz)
    T4_csr_mat = fbd.get_csr_matrix_scalar_op(He4_lookup, myTkin, nstat)
    V4_csr_mat = fbd.get_csr_matrix_scalar_op(He4_lookup, mycontact, nstat)
    W4_csr_mat = fbd.get_csr_matrix_scalar_op(He4_lookup, my3body, nstat)

    H4_csr_mat = T4_csr_mat + V4_csr_mat + W4_csr_mat
    
    k_eig=1
    vals, vecs = arpack_eigsh(H4_csr_mat, k=k_eig, which='SA')
    he4_fci_ens.append(np.round(vals[0] * phys_unit, 2))

    #CCSD
    #Compute 3He
    refEn, fock_mats, two_body_int = ccm.get_norm_ordered_ham(
    myL, ref.ref_3He_gs, myTkin, mycontact, my3body, sparse=True, NO2B=True)

    corrEn, t1, t2 = ccm.ccsd_solver(fock_mats, two_body_int, eps = 1.e-8, maxSteps = 100, 
                                max_diis = 10)
    
    he3_ccsd_ens.append(np.round((corrEn + refEn) * phys_unit, 2))

    #Compute 4He
    refEn, fock_mats, two_body_int = ccm.get_norm_ordered_ham(
    myL, ref.ref_4He_gs, myTkin, mycontact, my3body, sparse=True, NO2B=True)

    corrEn, t1, t2 = ccm.ccsd_solver(fock_mats, two_body_int, eps = 1.e-8, maxSteps = 100, 
                                max_diis = 10)
    
    he4_ccsd_ens.append(np.round((corrEn + refEn) * phys_unit, 2))

    
    #IMSRG(2)
    #Compute 3He
    if with_imsrg2:
        occs = normal_ordering.create_occupations(my_basis, ref.ref_3He_gs)
        e0, f, gamma = normal_ordering.compute_normal_ordered_hamiltonian_no2b(
                        occs, myTkin, mycontact, my3body
                    )
        e_imsrg, integration_data = ode_solver.solve_imsrg2(occs, e0, f, gamma, s_max=100, eta_criterion=1e-3)
        e_imsrg = np.round(e_imsrg * phys_unit, 2)
    else:
        e_imsrg = "-"
    he3_imsrg_ens.append(e_imsrg)

    #Compute 4He
    if with_imsrg2:
        occs = normal_ordering.create_occupations(my_basis, ref.ref_4He_gs)
        e0, f, gamma = normal_ordering.compute_normal_ordered_hamiltonian_no2b(
                        occs, myTkin, mycontact, my3body
                    )
        e_imsrg, integration_data = ode_solver.solve_imsrg2(occs, e0, f, gamma, s_max=100, eta_criterion=1e-3)
        e_imsrg = np.round(e_imsrg * phys_unit, 2)
    else:
        e_imsrg = "-"
    he4_imsrg_ens.append(e_imsrg)

#Table 5
print("".join(
    ["{:<14}".format(x) for x in ["a", "3He FCI", "3He IMSRG(2)", "3He CCSD", "4He FCI", "4He IMSRG(2)", "4He CCSD"]]
))
for i in range(3):
    print("".join(
        ["{:<14}".format(x) for x in 
         [params[i][0], he3_fci_ens[i], he3_imsrg_ens[i], he3_ccsd_ens[i], he4_fci_ens[i], he4_imsrg_ens[i], he4_ccsd_ens[i]]
        ]
    ))
