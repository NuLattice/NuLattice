import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent / ".."))
import NuLattice.lattice as lat
import NuLattice.references as ref
import NuLattice.CCM.coupled_cluster as ccm

import numpy as np

a_lat = 2.0
vS1 = -8
vT1 = vS1
w3 = 5.5
refs = [ref.ref_8Be_gs, ref.ref_12C_gs, ref.ref_16O_gs]

#solves for L=3,4, you can also add L=5 to the loop, but that will take a while to calculate
for thisL in [3,4]:
    for i in range(len(refs)):
        print(f'L={thisL} ',end='')
        ref_state = refs[i]
        if i == 0:
            print('8Be GS Energy: ', end = '')
        if i == 1:
            print('12C GS Energy: ', end = '')
        if i == 2:
            print('16O GS Energy: ', end = '')
        refEn, fock_mats, two_body_int = ccm.get_norm_ord_int(
                                            thisL, ref_state, vT1, vS1, w3)
        corrEn, t1, t2 = ccm.ccsd_solver(fock_mats, two_body_int, 
                            eps = 1e-8)
        phys_unit = lat.phys_unit(a_lat)
        gsEn = np.round((corrEn + refEn) * phys_unit, 2)
        print(f'{gsEn} MeV')