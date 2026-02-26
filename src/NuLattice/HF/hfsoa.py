"""
functions to perform a Hartree-Fock computation on the lattice
"""

__authors__ = "Thomas Papenbrock"
__credits__ = ["Thomas Papenbrock"]
__copyright__ = "(c) Thomas Papenbrock"
__license__ = "BSD-3-Clause"
__date__ = "2025-07-26"

import numpy as np
from typing import Tuple

try:
    from NuLattice._types import OneBodyOperator, TwoBodyOperator, ThreeBodyOperator
except ImportError:
    OneBodyOperator = TwoBodyOperator = ThreeBodyOperator = None

try:
    from numba import njit
except ImportError:
    print("Warning: Numba not detected. Some functions may run slower")

    def njit(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func

def contract_2nf(v2: TwoBodyOperator, dens: np.ndarray) -> np.ndarray:
    """
    takes list of two-body matrix elements and contracts them with the density to get a one-body operator

    :param v2:   TwoBodyOperator object
    :type v2:    list[list[int,int,int,int, float]] | TwoBodyOperator
    :param dens: square density matrix
    :type dens:  numpy.array((:,:), dtype=float)
    :return:     one-body operator of the same shape as the density matrix dens
    :rtype:      numpy.array((:,:), dtype=float)
    """
    n_states = dens.shape[0]
    res = np.zeros((n_states, n_states), dtype=dens.dtype)
    _contract_2nf_kernel(v2.indices, v2.values, dens, res)
    return res


@njit
def _contract_2nf_kernel(indices, values, dens, res):
    for i in range(len(values)):
        pi, qi, ri, si = indices[i]
        v = values[i]

        res[pi, ri] += v * dens[qi, si]
        res[qi, ri] -= v * dens[pi, si]
        res[pi, si] -= v * dens[qi, ri] # Note: this is dens index r/s swap
        res[qi, si] += v * dens[pi, ri]

def contract_3nf(w3, dens):
    """
    takes list of three-body matrix elements and contracts them with the density to get a one-body operator

    :param w3:   list of two-body matrix elements [p,q,r,s,value]
                 OR ThreeBodyOperator object
    :type w3:    list[list[int,int,int,int,int,int, float]] | ThreeBodyOperator
    :param dens: square density matrix
    :type dens:  numpy.array((:,:), dtype=float)
    :return:     one-body operator of the same shape as the density matrix dens
    :rtype:      numpy.array((:,:), dtype=float)
    """
    return _contract_3nf_soa(w3, dens)

def _contract_3nf_soa(w3_op, dens):
    """
    Optimized pass-through for ThreeBodyOperator.
    Passes contiguous internal arrays directly to the Numba kernel.
    """
    n_states = dens.shape[0]
    res = np.zeros((n_states, n_states), dtype=dens.dtype)
    _contract_3nf_kernel(w3_op.indices, w3_op.values, dens, res)
    return res


@njit
def _contract_3nf_kernel(w3_indices, w3_vals, dens, res):
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

def init_density(nstat: int, hole: Tuple[int]):
    """
    creates a density matrix of dimension nstat x nstat given the hole information

    :param nstat: dimension of single-particle basis
    :type nstat:  int
    :param hole:  tuple of occupied single-particle states, as numbers from 0 ... A-1
    :type hole:   tuple(int, int, ... )
    :return:      density matrix where hole states are occupied (1) and all others not (0)
    :rtype:       numpy.array((nstat,nstat), dtype = float)
    """
    dens = np.zeros((nstat, nstat))
    for i in hole:
        dens[i, i] = 1.0
    return dens


def HF_iter(
    op1: OneBodyOperator,
    op2: TwoBodyOperator,
    op3: ThreeBodyOperator,
    dens: np.ndarray,
    mix: float = 0.5,
):
    """
    Performs one iteration of the Hartree-Fock procedure
    """
    return _HF_iter(op1, op2, op3, dens, mix=0.5)

def _HF_iter_ref(
    h1: np.ndarray,
    op2: TwoBodyOperator,
    op3: ThreeBodyOperator,
    dens: np.ndarray,
    mix: float,
    gamma_buf: np.ndarray,
    omega_buf: np.ndarray,
    hf_ham_buf: np.ndarray,
    dens_buf: np.ndarray
):
    npart = int(round(np.trace(dens)))
    
    gamma_buf.fill(0.0)
    omega_buf.fill(0.0)
    
    _contract_2nf_kernel(op2.indices, op2.values, dens, gamma_buf)
    _contract_3nf_kernel(op3.indices, op3.values, dens, omega_buf)

    np.copyto(hf_ham_buf, h1)
    hf_ham_buf += gamma_buf
    hf_ham_buf += 0.5 * omega_buf

    # Compute Energy
    # E_op = h + 0.5 * Gamma + 1/6 * Omega
    # E = Tr( (h + 0.5*Gamma + 1/6*Omega) * rho )
    #   = Tr(h rho) + 0.5*Tr(Gamma rho) + 1/6*Tr(Omega rho)
    e_h1 = np.sum(h1 * dens)
    e_gamma = np.sum(gamma_buf * dens)
    e_omega = np.sum(omega_buf * dens)
    
    erg = e_h1 + 0.5 * e_gamma + (1.0 / 6.0) * e_omega

    # NOTE(vivek): new bottleneck
    vals, vecs = np.linalg.eigh(hf_ham_buf)

    # Select occupied orbitals (first npart columns)
    occ = vecs[:, :npart]
    
    # new_dens = occ @ occ.T.
    # O(N^3) BLAS
    np.matmul(occ, occ.T, out=dens_buf)


    dens_buf -= dens
    diff_dens = np.sum(np.abs(dens_buf))

    if mix != 0:
        dens_buf *= mix
    dens += dens_buf

    return erg, diff_dens, vecs

def _HF_iter(
    h1: np.ndarray,
    op2: TwoBodyOperator,
    op3: ThreeBodyOperator,
    dens: np.ndarray,
    mix=0.5,
):
    npart = int(round(np.trace(dens)))

    # nstat = dens.shape[0]
    # h1 = op1.to_dense(nstat)
    gamma = contract_2nf(op2, dens)
    omega = contract_3nf(op3, dens)

    e_op = h1 + 0.5 * gamma + (1.0 / 6.0) * omega
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


def solve_HF(
    op1: OneBodyOperator,
    op2: TwoBodyOperator,
    op3: ThreeBodyOperator,
    dens: np.ndarray,
    mix: float = 0.5,
    eps: float = 1.0e-8,
    max_iter: int = 100,
    verbose: bool = False,
):
    """
    Solve the Hartree-Fock problem using Zero-Copy strategy.
    """
    converged = False
    
    nstat = dens.shape[0]
    _dens = dens.copy() 
    gamma_buf = np.zeros((nstat, nstat), dtype=dens.dtype)
    omega_buf = np.zeros((nstat, nstat), dtype=dens.dtype)
    ham_buf = np.zeros((nstat, nstat), dtype=dens.dtype)
    dens_buf = np.zeros((nstat, nstat), dtype=dens.dtype)
    
    h1_dense = op1.to_dense()

    erg0 = 0.0

    for i in range(max_iter):
        erg, diff_dens, vecs = _HF_iter_ref(
            h1_dense, 
            op2, 
            op3, 
            _dens, 
            mix,
            gamma_buf,
            omega_buf,
            ham_buf,
            dens_buf
        )
        
        diff = abs(erg - erg0)
        if verbose:
             print(f"Iter {i}: E={erg:.8f}, dE={diff:.6e}")

        if diff_dens < eps and i > 1:
            converged = True
            break
            
        erg0 = erg
        
    return erg, vecs, converged
