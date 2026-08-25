"""
This module provides functions to define two body operators on the 3D lattice   
"""
__authors__   =  ["Thomas Papenbrock, Maxwell Rothman"]
__credits__   =  ["Thomas Papenbrock, Maxwell Rothman"]
__copyright__ = "(c) Thomas Papenbrock, Maxwell Rothman" 
__license__   = "BSD-3-Clause"
__date__      = "2026-05-15"


import numpy as np
import itertools
import scipy.sparse as sparse
import sys
import os
from concurrent.futures import ProcessPoolExecutor

import NuLattice.lattice as lat
import NuLattice.constants as consts
import NuLattice.operators.one_body_operators as ob_ops

def rho_mult_NO(rho_1, rho_2, mult, max_mem = 0):
    """
    Multiplies the given density operators normal orders the product
    
    :param rho_1:   Array of first density operator
    :type rho_1:    scipy.sparse.coo_array
    :param rho_2:   Array of second density operator
    :type rho_2:    scipy.sparse.coo_array
    :param mult:    Factor to scale the calculation by
    :type mult:     float
    :param max_mem: Optional; Maximum memory size for the temporary arrays
                    to get to before compressing into a sparse format.
                    If left as 0, no limit to the memory will be set
    :type max_mem:  int
    :returns:       The density operators multilpied and normal ordered,
                    for V^{ij}_{kl}, it only stores i<j and k<l
    :rtype:         scipy.sparse.csr_array
    """
    tmp_col = []
    tmp_row = []
    tmp_val = []   
    dim = rho_1.shape[0]
    ret = sparse.csr_array((dim ** 2, dim ** 2))    
    for a, c, v in zip(rho_1.row, rho_1.col, rho_1.data):
        for b, d, w in zip(rho_2.row, rho_2.col, rho_2.data):                
            matele = mult * v * w
            if abs(np.real_if_close(matele)) < 1e-15:
                continue
            if a < b and c < d:
                tmp_col.append(a + b * dim)
                tmp_row.append(c + d * dim)
                tmp_val.append(matele)
            if b < a and c < d:
                tmp_col.append(b + a * dim)
                tmp_row.append(c + d * dim)
                tmp_val.append(-matele)
                if max_mem != 0 and sys.getsizeof(tmp_val) + sys.getsizeof(tmp_col) + sys.getsizeof(tmp_row) > max_mem:
                    ret += sparse.csr_array((tmp_val, (tmp_row, tmp_col)), shape = (dim ** 2, dim ** 2))
                    tmp_val = []
                    tmp_row = []
                    tmp_col = []

    return ret + sparse.csr_array((tmp_val, (tmp_row, tmp_col)), shape = (dim ** 2, dim ** 2))


def shortRangeV_2body(lattice, myL, sL, sNL, c0, op1b = None, spin = 2, isospin = 2, verbose = False, max_mem = 0, sites=None):
    """
    Generates a short range two body interaction of the form sum_n :rho(n):^2 where rho is a smeared density
    
    :param lattice: list of lattice sites returned by lattice.get_lattice
    :type lattice:  list[(int, int, int)] 
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param sL:      local smearing strength
    :type sL:       float
    :param sNL:     non-local smearing strength
    :type sNL:      float
    :param c0:      strength of the interaction in MeV
    :type c0:       float
    :param op1b:    Optional; one body operator used to generate rho in the form 
                    a^dagger [op1b] a. If None, then the identity operator is used
    :type op1b:     scipy.sparse.csr_array()
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int  
    :param verbose: Optional; whether or not to print progress during calculation
    :type verbose:  bool
    :param max_mem: Optional; maximum memory for all of the temporary float arrays to take
                    up before compressing into a sparse format. If set to 0, it will completely
                    fill the array before compressing. NOTE: this utilizes a multiprocess to
                    parallelize the loop over all sites, so the memory for each site will be
                    max_mem / cpu_count and you should set the memory limit accordingly
    :type max_mem:  float
    :param sites:   Optional; Give default value or None in order to compute the interaction at 
                    all sites, or give a list of sites in the format [i, j, k] to only compute it 
                    at the given sites
    :type sites:    list[int,int,int]
    :return:        A sparse csr_array representing V^{ab}_{ij} in MeV, 
                    for dim=spin*isospin*L^3, the row indicies correspond
                    to a + b * dim and column indices correspond to i + j * dim
    :rtype:         scipy.sparse.csr_array()
    """
    rho_smeared = ob_ops.get_smeared_dens(lattice, myL, sL, sNL, op1b = op1b, spin=spin, isospin=isospin, verbose=verbose, sites=sites)
    dim  = myL **3 * spin * isospin
    if verbose:
        print('Generating Interaction...',end='')
    ret = sparse.csr_array((dim ** 2, dim ** 2))
    with ProcessPoolExecutor() as executor:
        max_mem_proc = max_mem / os.cpu_count()
        size = len(rho_smeared)
        for val in executor.map(rho_mult_NO, rho_smeared, rho_smeared, [c0] * size, [max_mem_proc] * size):
            ret += val
    if verbose:
        print('Interaction Generated')

    return ret

def f_SS(myL, bpi, a_lat, m_pi_0 = consts.m_pi_0):
    """
    f_S'S function to be used in one pion exchange
    
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param bpi:     Parameter to remove the short-distance lattice artifacts
    :type bpi:      float
    :param a_lat:   lattice spacing divided by hbar c
    :type a_lat:    float
    :param m_pi_0:  Optional; Neutral pion mass
    :type m_pi_0:   float
    :return:        f_S'S(m), where m is the vector n'-n, as a numpy array that gets 
                    indexed as fSS[S'][S][m_x][m_y][m_z]
    :rtype:         complex numpy array of dimension 3,3, myL, myL, myL
    """
    m_pi = m_pi_0 * a_lat
    q =np.fft.fftfreq(myL, 1/ (2.0 * np.pi))
    qx, qy, qz = np.meshgrid(q, q, q, indexing="ij")

    q2 = qx**2 + qy**2 + qz**2
    ft_fss = np.exp(-bpi * q2) / (q2 + m_pi**2)

    q = np.array([qx, qy, qz])

    fSS = np.zeros((3, 3, myL, myL, myL), dtype=complex)

    for s1, s2 in itertools.product(range(3), range(3)):
        fSS[s1, s2] = np.fft.ifftn(q[s1] * q[s2] * ft_fss)

    return fSS

def onePionEx(myL, bpi, a_lat, lattice, verbose = False, mult = 1,g_A=consts.g_A, f_pi = consts.f_pi, m_pi_0 = consts.m_pi_0, max_mem=1e9):
    """
    computes the potential for one pion exchange

    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param bpi:     parameter to remove short-distance lattice artifacts
    :type bpi:      float
    :param a_lat:   lattice spacing divided by hbar c
    :type a_lat:    float
    :param lattice: list of lattice sites returned by lattice.get_lattice
    :type lattice:  list[(int, int, int)] 
    :param verbose: Optional; whether or not to print progress during calculation
    :type verbose:  bool
    :param mult:    Optional; scale factor to multiply potential by
    :type mult:     float
    :param g_A:     Optional; Axial-vector coupling constant
    :type g_A:      float
    :param f_pi:    Optional; Pion decay constant
    :type f_pi:     float
    :param m_pi_0:  Optional; Neutral pion mass
    :type m_pi_0:   float
    :param max_mem: maximum memory size for the temporary float array to get to 
                    before compressing it into a scipy.sparse array
    :type max_mem:  float
    :return:        A sparse csr_array representing V^{ab}_{ij} in MeV with the 
                    row indicies corresponding to a + b * 4*L^3 and 
                    column indices corresponding to i + j * 4*L^3
    :rtype:         scipy.sparse.csr_array()
    """
    if verbose:
        print('Calculating f_SS...', end='')
    fSS = f_SS(myL, bpi, a_lat, m_pi_0=m_pi_0)
    scale = - (g_A / (2.0 * a_lat * f_pi)) ** 2 * mult / 2.0
    dim  = myL **3 * 4
    if verbose:
        print('Done\nCalculating Densities...', end='')
    dens = [None] * myL ** 3
    sp_ops = [ob_ops.list_to_sparse1b(ob_ops.spin_x(lattice, myL)), 
            ob_ops.list_to_sparse1b(ob_ops.spin_y(lattice, myL)), 
            ob_ops.list_to_sparse1b(ob_ops.spin_z(lattice, myL))]
    iso_ops = [ob_ops.list_to_sparse1b(ob_ops.tau_x(lattice, myL)), 
           ob_ops.list_to_sparse1b(ob_ops.tau_y(lattice, myL)), 
           ob_ops.list_to_sparse1b(ob_ops.tau_z(lattice, myL))]
    for site in lattice:
        rho_sp = [None] * 3
        for sp in range(3):
            rho_sp_iso = [None] * 3
            for iso in range(3):
                op1b = sp_ops[sp] @ iso_ops[iso]
                rho_sp_iso[iso] =ob_ops.rho_op(site, myL, op1b,op_fac = 4.0, spin=2, isospin=2)
            rho_sp[sp] = rho_sp_iso
        loc = lat.site2index(site, myL)
        dens[loc] = rho_sp
    sparse_full_ope = sparse.csr_array((dim * dim, dim * dim))
    tmp_val = []
    tmp_row = []
    tmp_col = []
    if verbose:
        print('Done\nGenerating Interaction...')

    for sp1, sp2, iso, site1, site2 in itertools.product(range(3), range(3), 
                                                         range(3), lattice, lattice):
        f_SS_val = fSS[sp1, sp2, (site1[0] - site2[0]) % myL, (site1[1] - site2[1]) % myL, 
                  (site1[2] - site2[2]) % myL]
        if f_SS_val == 0:
            continue
        loc1 = lat.site2index(site1, myL)
        loc2 = lat.site2index(site2, myL)
        rho1 = dens[loc1][sp1][iso]
        rho2 = dens[loc2][sp2][iso]
        
        for a, c, v in zip(rho1.row, rho1.col, rho1.data):
            for b, d, w in zip(rho2.row, rho2.col, rho2.data):     
                matele = f_SS_val * v * w * scale / a_lat
                if abs(np.real_if_close(matele)) < 1e-15:
                    continue
                if a < b and c < d:
                    tmp_col.append(a + b * dim)
                    tmp_row.append(c + d * dim)
                    tmp_val.append(matele)
                if b < a and c < d:
                    tmp_col.append(b + a * dim)
                    tmp_row.append(c + d * dim)
                    tmp_val.append(-matele)
                if a > b and c > d:
                    tmp_col.append(b + a * dim)
                    tmp_row.append(d + c * dim)
                    tmp_val.append(matele)
                if b > a and c > d:
                    tmp_col.append(a + b * dim)
                    tmp_row.append(d + c * dim)
                    tmp_val.append(-matele)
                if max_mem != 0 and sys.getsizeof(tmp_val) > max_mem:
                    if verbose:
                        print(f'Compressing interaction...',end='')
                    sparse_full_ope += sparse.csr_array((tmp_val, (tmp_row, tmp_col)), shape = (dim * dim, dim * dim))
                    tmp_val = []
                    tmp_row = []
                    tmp_col = []
                    if verbose:
                        print('Compressed')
    if len(tmp_col) > 0:
        if verbose:
            print(f'Compressing interaction...',end='')
        sparse_full_ope += sparse.csr_array((tmp_val, (tmp_row, tmp_col)), shape = (dim * dim, dim * dim))
        if verbose:
            print('Done')
    if verbose:
        print('Interaction Generated')

    return sparse_full_ope

def sparse_to_list_2body(sparse_int, myL, spin=2, isospin=2):
    """
    Takes sparse two-body interaction and converts it to a list

    :param sparse_int:  Interaction stored as a sparse matrix
    :type sparse_int:   scipy.sparse.csr_array
    :param myL:         number of lattice sites in each direction
    :type myL:          int
    :param spin:        Optional; number of spin degrees of freedom
    :type spin:         int
    :param isospin:     Optional; number of isospin degrees of freedom
    :type isospin:      int  
    :return:            list of lists [a,b,c,d,v] where a, b, c, and d are 
                        indices and v is the value for V^ab_cd
    :rtype:             list[[int, int, int, int, float]]
    """
    ret = []
    dim  = myL ** 3 * spin * isospin
    sparse_int_coo = sparse_int.tocoo()
    for i, j, v in zip(sparse_int_coo.row, sparse_int_coo.col, sparse_int_coo.data):
        a = i % dim
        b = (i - a) // dim
        c = j % dim
        d = (j - c) // dim

        ret.append([a, b, c, d, v])
    return ret

def change_lat_2body(inter, origL, newL, spin=2, isospin=2):
    """
    Changes a two-body interaction in list format for a given L to a new L

    :param inter:   interaction stored as a list of lists [a,b,c,d,v] 
                    where a, b, c, and d are indices and v is the value 
                    for V^ab_cd
    :type inter:    list[(int, int, int, int, float)]
    :param origL:   original L for the basis of inter
    :type origL:    int
    :param newL:    new L to return the basis of inter
    :type newL:     int
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int
    :return:        interaction in the basis of the new L in the same
                    list format
    :rtype:         list[(int, int, int, int, float)]    
    """
    new_inter = [[] for _ in range(len(inter))]
    for i in range(len(inter)):
        a, b, c, d, val = inter[i]
        lst = []
        ind_lst = [a,b,c,d]
        for ind in ind_lst:
            lst.append(lat.state2index(lat.index2state(ind, origL, spin, isospin), newL, spin, isospin))
        lst.append(val)
        new_inter[i] = lst
    return new_inter