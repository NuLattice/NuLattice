"""
functions to perform a Hartree-Fock computation on the lattice
"""
__authors__   =  "Thomas Papenbrock"
__credits__   =  ["Thomas Papenbrock"]
__copyright__ = "(c) Thomas Papenbrock"
__license__   = "BSD-3-Clause"
__date__      = "2025-07-26"

import numpy as np
from functools import wraps

try:
    from numba import njit
except ImportError:
    print("Warning: Numba not detected. Some functions may run slower")
    def njit(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func

from opt_einsum import contract

def _cache_v2_matrix(func):
    """
    Custom decorator to cache the prepared v2 matrix 
    based on the memory address of the v2 interaction list.
    """
    cache = {}

    @wraps(func)
    def wrapper(v2, dens):
        v2_id = id(v2)
        # NOTE(vivek): Assumes the interaction is constant during the HF flow
        if v2_id not in cache:
            cache[v2_id] = _prepare_v2_matrix(v2, dens.shape[0])
        return func(cache[v2_id], dens)
    
    return wrapper

def get_1body_matrix(myTkin, nstat: int) -> np.ndarray:
    """
    takes the list of one-body matrix elements and turns it into a square matrix
    
    :param nstat:  dimension of matrix, i.e. the number of 1-body states
    :type nstat:   int
    :param myTkin: list of one-body matrix elements [[p1,q1,value1], [p2,q2,value2], ...]
    :type myTkin:  list[list[int,int, float]]
    :return:       nstat x nstat matrix of the list of matrix elements
    :rtype:        numpy.array((:,:), dtype=float)
    """
    return _get_1body_matrix_np(myTkin, nstat)

def _get_1body_matrix_original(myTkin, nstat):
    op1 = np.zeros((nstat,nstat))
    for [a, b, val] in myTkin:
        op1[a,b]=val
    return op1

def _get_1body_matrix_np(myTkin,nstat):
    op1 = np.zeros((nstat, nstat))
    arr = np.array(myTkin)
    indices_a = arr[:, 0].astype(int)
    indices_b = arr[:, 1].astype(int)
    values = arr[:, 2]
    op1[indices_a, indices_b] = values
    return op1


def contract_2nf(v2,dens):
    """
    takes list of two-body matrix elements and contracts them with the density to get a one-body operator

    :param v2:   list of two-body matrix elements [p,q,r,s,value] 
    :type v2:    list[list[int,int,int,int, float]]
    :param dens: square density matrix
    :type dens:  numpy.array((:,:), dtype=float)
    :return:     one-body operator of the same shape as the density matrix dens
    :rtype:      numpy.array((:,:), dtype=float)
    """
    return _contract_2nf_original(v2, dens)

def _contract_2nf_original(v2, dens):
    res = np.zeros_like(dens)
    for mat_ele in v2:
        [a, b, c, d, val] = mat_ele
        
        res[a,c] += val*dens[b,d] #1
        res[b,c] -= val*dens[a,d] #P(ab)
        res[a,d] -= val*dens[b,c] #P(cd)
        res[b,d] += val*dens[a,c] #P(ab)P(cd) 
    return res


def _prepare_v2_matrix(v2, nstat):
    """Bakes antisymmetry into a 2D matrix (N^2, N^2)."""
    V = np.zeros((nstat, nstat, nstat, nstat))
    for a, b, c, d, val in v2:
        V[a, b, c, d] += val
        V[b, a, c, d] -= val
        V[a, b, d, c] -= val
        V[b, a, d, c] += val
    return V.transpose(0, 2, 1, 3).reshape(nstat**2, nstat**2)

@_cache_v2_matrix
def _contract_2nf_np(v2_matrix, dens):
    """
    Perform the O(N^3) contraction.
    Note: The decorator passes the cached MATRIX here, not the list!
    """
    dim = dens.shape[0]
    # (N^2, N^2) @ (N^2, 1)
    res_flat = v2_matrix @ dens.ravel()
    return res_flat.reshape(dim, dim)

def contract_3nf(w3, dens):
    """
    takes list of three-body matrix elements and contracts them with the density to get a one-body operator

    :param w3:   list of two-body matrix elements [p,q,r,s,value] 
    :type w3:    list[list[int,int,int,int,int,int, float]]
    :param dens: square density matrix
    :type dens:  numpy.array((:,:), dtype=float)
    :return:     one-body operator of the same shape as the density matrix dens
    :rtype:      numpy.array((:,:), dtype=float)
    """
    return _contract_3nf_np(w3, dens)

def _contract_3nf_original(w3,dens):
    res = np.zeros_like(dens)
    for mat_ele in w3:  # we need all 36 antisymmetric combinations of the ket (abc) and bra (def) single-particle states
        [a, b, c, d, e, f, val] = mat_ele
        res[a,d] += val*( dens[b,e]*dens[c,f]  # (abc), (def), antisym last two pairs
                         -dens[c,e]*dens[b,f]
                         -dens[b,f]*dens[c,e]
                         +dens[c,f]*dens[b,e] )
        res[b,d] += val*( dens[c,e]*dens[a,f]  # (bca), (def), antisym last two pairs
                         -dens[a,e]*dens[c,f]
                         -dens[c,f]*dens[a,e]
                         +dens[a,f]*dens[c,e] )        
        res[c,d] += val*( dens[a,e]*dens[b,f]  # (cab), (def), antisym last two pairs
                         -dens[b,e]*dens[a,f]
                         -dens[a,f]*dens[b,e]
                         +dens[b,f]*dens[a,e] )
        res[a,e] += val*( dens[b,f]*dens[c,d]  # (abc), (efd), antisym last two pairs
                         -dens[c,f]*dens[b,d]
                         -dens[b,d]*dens[c,f]
                         +dens[c,d]*dens[b,f] )
        res[b,e] += val*( dens[c,f]*dens[a,d]  # (bca), (efd), antisym last two pairs
                         -dens[a,f]*dens[c,d]
                         -dens[c,d]*dens[a,f]
                         +dens[a,d]*dens[c,f] )        
        res[c,e] += val*( dens[a,f]*dens[b,d]  # (cab), (efd), antisym last two pairs
                         -dens[b,f]*dens[a,d]
                         -dens[a,d]*dens[b,f]
                         +dens[b,d]*dens[a,f] )
        res[a,f] += val*( dens[b,d]*dens[c,e]  # (abc), (fde), antisym last two pairs
                         -dens[c,d]*dens[b,e]
                         -dens[b,e]*dens[c,d]
                         +dens[c,e]*dens[b,d] )
        res[b,f] += val*( dens[c,e]*dens[a,f]  # (bca), (fde), antisym last two pairs
                         -dens[a,e]*dens[c,f]
                         -dens[c,f]*dens[a,e]
                         +dens[a,f]*dens[c,e] )        
        res[c,f] += val*( dens[a,d]*dens[b,e]  # (cab), (fde), antisym last two pairs
                         -dens[b,d]*dens[a,e]
                         -dens[a,e]*dens[b,d]
                         +dens[b,e]*dens[a,d] )
    return res

@njit
def _contract_3nf_kernel(w3_indices, w3_vals, dens):
    nstat = dens.shape[0]
    res = np.zeros((nstat, nstat))
    
    for i in range(len(w3_vals)):
        a, b, c, d, e, f = w3_indices[i]
        val = w3_vals[i]
        
        rbe = dens[b, e]
        rcf = dens[c, f]
        rce = dens[c, e]
        rbf = dens[b, f]
        
        rae = dens[a, e]
        raf = dens[a, f]
        
        rbd = dens[b, d]
        rcd = dens[c, d]
        rad = dens[a, d]
        
        # res[?, d]
        res[a, d] += val * 2.0 * (rbe * rcf - rce * rbf)
        res[b, d] += val * 2.0 * (rce * raf - rae * rcf)
        res[c, d] += val * 2.0 * (rae * rbf - rbe * raf)
        
        # res[?, e]
        res[a, e] += val * 2.0 * (rbf * rcd - rcf * rbd)
        res[b, e] += val * 2.0 * (rcf * rad - raf * rcd)
        res[c, e] += val * 2.0 * (raf * rbd - rbf * rad)
        
        # res[?, f]
        res[a, f] += val * 2.0 * (rbd * rce - rcd * rbe)
        res[b, f] += val * 2.0 * (rce * raf - rae * rcf)
        res[c, f] += val * 2.0 * (rad * rbe - rbd * rae)
        
    return res

def _contract_3nf_np(w3, dens):
    w3_arr = np.array(w3)
    w3_indices = w3_arr[:, :6].astype(np.int32)
    w3_vals = w3_arr[:, 6]
    return _contract_3nf_kernel(w3_indices, w3_vals, dens)

def make_HF_ham(op1,op2,op3,dens):
    """
    takes Hamiltonian consisting of one-body operator op1, two-body operator op2,
    and three-body operator op3, and the density matrix and returns the Hartree-Fock Hamiltonian.

    :param op1:  list of one-body matrix elements
    :type op1:   list[list[int,int, float]]
    :param op2:  list of two-body matrix elements
    :type op2:   list[list[int,int,int,,int, float]]
    :param op3:  list of three-body matrix elements
    :type op3:    list[list[int,int,int,int,int,int, float]]
    :param dens: density matrix (same shape as op1)
    :type dens:  numpy.array((:,:), dtype=float)
    :return:     matrix in the shape of op1 and dens that is the Hartree-Fock Hamiltonian
    :rtype:      numpy.array((:,:), dtype=float)
    """
    nstat = len(dens)
    hf_op = get_1body_matrix(op1,nstat)
    hf_op += contract_2nf(op2,dens)
    hf_op += 0.5*contract_3nf(op3,dens)
    return hf_op

def init_density(nstat,hole):
    """
    creates a density matrix of dimension nstat x nstat given the hole information

    :param nstat: dimension of single-particle basis
    :type nstat:  int
    :param hole:  tuple of occupied single-particle states, as numbers from 0 ... A-1
    :type hole:   tuple(int, int, ... )
    :return:      density matrix where hole states are occupied (1) and all others not (0)
    :rtype:       numpy.array((nstat,nstat), dtype = float)
    """
    dens = np.zeros((nstat,nstat))
    for i in hole:
        dens[i,i] = 1.0
    return dens

# def _init_density_np(nstat, hole):
#     dens = np.zeros((nstat,nstat))
#     dens[list(hole), list(hole)] = 1.0
#     return dens


def HF_energy(op1, op2, op3, dens):
    """
    Computes the Hartree-Fock energy for a given density dens and Hamiltonian consisting
    of one-body term op1, two-body term op2, and three-body term op3

    :param op1:  list of one-body matrix elements
    :type op1:   list[list[int,int, float]]
    :param op2:  list of two-body matrix elements
    :type op2:   list[list[int,int,int,int, float]]
    :param op3:  list of three-body matrix elements
    :type op3:   list[list[int,int,int,int,int,int, float]]
    :param dens: density matrix (same shape as op1)
    :type dens:  numpy.array((:,:), dtype=float)
    :return:     Hartree-Fock energy
    :rtype:      float
    """
    nstat = len(dens)
    dum = get_1body_matrix(op1,nstat)
    dum += 0.5*contract_2nf(op2,dens)
    dum += (1.0/6.0)*contract_3nf(op3,dens)
    erg = np.sum(dum * dens.T)
    return erg

def HF_iter(op1, op2, op3, dens, mix=0.5):
    """
    Performs one iteration of the Hartree-Fock procedure

    :param op1:  list of one-body matrix elements
    :type op1:   list[list[int,int, float]]
    :param op2:  list of two-body matrix elements
    :type op2:   list[list[int,int,int,,int, float]]
    :param op3:  list of three-body matrix elements
    :type op3:   list[list[int,int,int,int,int,int, float]]
    :param dens: density matrix (same shape as op1)
    :type dens:  numpy.array((:,:), dtype=float)
    :param mix:  returned density matrix is mix*new_density + (1-mix)*old_density
    :type mix:   float 
    :return:     energy, density, vecs as the current HF energy, current density
                 matrix, and orthogonal transformation matrix that diagonalizes
                 the HF Hamiltonian
    :rtype:      float, numpy.array((:,:), dtype=float), numpy.array((:,:), dtype=float)
    """
    return _HF_iter_np(op1, op2, op3, dens, mix=0.5)

def _HF_iter_original(op1, op2, op3, dens, mix=0.5):
    npart=round(np.trace(dens)) # rounds to nearest integer
    erg = HF_energy(op1, op2, op3, dens)
    hf = make_HF_ham(op1, op2, op3, dens)
    vals, vecs = np.linalg.eigh(hf)
    new_dens=contract("pi,qi->pq", vecs[:,0:npart], vecs[:,0:npart])
    res_dens = mix*new_dens + (1.0-mix)*dens
    return erg, res_dens, vecs

def _HF_iter_np(op1, op2, op3, dens, mix=0.5):
    npart = int(round(np.trace(dens)))
    
    nstat = dens.shape[0]
    h1 = get_1body_matrix(op1, nstat)
    gamma = contract_2nf(op2, dens)
    omega = contract_3nf(op3, dens)
    
    e_op = h1 + 0.5 * gamma + (1.0/6.0) * omega
    erg = np.dot(e_op.ravel(), dens.T.ravel())
    
    hf_ham = h1 + gamma + 0.5 * omega
    
    vals, vecs = np.linalg.eigh(hf_ham)
    
    # 'pi,qi->pq' einsum
    occ = vecs[:, :npart]
    new_dens = occ @ occ.T
    
    if mix != 1.0:
        res_dens = mix * new_dens + (1.0 - mix) * dens
    else:
        res_dens = new_dens
        
    return erg, res_dens, vecs


def solve_HF(op1, op2, op3, dens, mix=0.5, eps=1.e-8, max_iter=100, verbose=False):
    """
    Solve the Hartree-Fock problem

    :param op1:  list of one-body matrix elements
    :type op1:   list[list[int,int, float]]
    :param op2:  list of two-body matrix elements
    :type op2:   list[list[int,int,int,,int, float]]
    :param op3:  list of three-body matrix elements
    :type op3:   list[list[int,int,int,int,int,int, float]]
    :param dens: density matrix (same shape as op1)
    :type dens:  numpy.array((:,:), dtype=float)
    :param mix:  parameter used in the mixing: mix*new_density + (1-mix)*old_density
    :type mix:   float
    :param eps:  converegence of energy
    :type eps:   float
    :param max_iter:  maximum number of HF iterations
    :type max_iter:   float
    :return:     energy, orthogonal transformation matrix that diagonalizes
                 the HF Hamiltonian (the first A columns are occupied), converged
    :rtype:      float, numpy.array((:,:), dtype=float), boolean
    """
    converged = False
    my_dens=dens.copy()
    erg0 = 0
    for i in range(max_iter):
        erg, new_dens, vecs = HF_iter(op1, op2, op3, my_dens, mix)
        diff = np.abs(erg-erg0)
        diff_dens = np.sum(np.abs(new_dens-my_dens))
        if verbose:
            print(i, "E=", erg, ", Delta E=", diff, ", Delta rho =", diff_dens)
        if diff_dens < eps and i > 1:
            converged = True
            break
        else:
            erg0 = erg
            my_dens = new_dens.copy()
    return erg, vecs, converged
