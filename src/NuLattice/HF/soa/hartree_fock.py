"""
functions to perform a Hartree-Fock computation on the lattice
"""

__authors__ = "Thomas Papenbrock"
__credits__ = ["Thomas Papenbrock"]
__copyright__ = "(c) Thomas Papenbrock"
__license__ = "BSD-3-Clause"
__date__ = "2025-07-26"

import torch
from typing import Tuple

try:
    from NuLattice._types import OneBodyOperator, TwoBodyOperator, ThreeBodyOperator
except ImportError:
    OneBodyOperator = TwoBodyOperator = ThreeBodyOperator = None

def contract_2nf(v2: TwoBodyOperator, dens: torch.tensor) -> torch.tensor:
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
    res = torch.zeros((n_states, n_states), dtype=dens.dtype, device=dens.device)
    _contract_2nf_kernel(v2.indices, v2.values, dens, res)
    
    
    return res

def _contract_2nf_kernel(op2_indices, op2_vals, dens, res):
    # Extract columns of indices: shape (N, 4) -> four (N,) tensors
    p, q, r, s = op2_indices[:, 0], op2_indices[:, 1], op2_indices[:, 2], op2_indices[:, 3]
    v = op2_vals

    # Bulk updates using index_put_ with accumulate=True
    # This replaces the entire Python loop
    res.index_put_((p, r), v * dens[q, s], accumulate=True)
    res.index_put_((q, r), -v * dens[p, s], accumulate=True)
    res.index_put_((p, s), -v * dens[q, r], accumulate=True)
    res.index_put_((q, s), v * dens[p, r], accumulate=True)


@torch.compile
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
    n_states = dens.shape[0]
    res = torch.zeros((n_states, n_states), dtype=dens.dtype, device=dens.device)
    _contract_3nf_kernel(w3.indices, w3.vals, dens, res)
    return res

@torch.compile
def _contract_3nf_kernel(w3_indices, w3_vals, dens, res):
    # w3_indices shape: (N, 6)
    a, b, c, d, e, f = [w3_indices[:, i] for i in range(6)]
    val = w3_vals

    # Pre-fetch density elements (Vectorized)
    rbe, rcf = dens[b, e], dens[c, f]
    rce, rbf = dens[c, e], dens[b, f]
    rae, raf = dens[a, e], dens[a, f]
    rbd, rcd, rad = dens[b, d], dens[c, d], dens[a, d]

    # Pre-calculate the scale
    v2 = val * 2.0

    # Accumulate into 'res' for each target index pair
    # res[?, d]
    res.index_put_((a, d), v2 * (rbe * rcf - rce * rbf), accumulate=True)
    res.index_put_((b, d), v2 * (rce * raf - rae * rcf), accumulate=True)
    res.index_put_((c, d), v2 * (rae * rbf - rbe * raf), accumulate=True)

    # res[?, e]
    res.index_put_((a, e), v2 * (rbf * rcd - rcf * rbd), accumulate=True)
    res.index_put_((b, e), v2 * (rcf * rad - raf * rcd), accumulate=True)
    res.index_put_((c, e), v2 * (raf * rbd - rbf * rad), accumulate=True)

    # res[?, f]
    res.index_put_((a, f), v2 * (rbd * rce - rcd * rbe), accumulate=True)
    res.index_put_((b, f), v2 * (rce * raf - rae * rcf), accumulate=True) # Check if this logic matches your kernel swap
    res.index_put_((c, f), v2 * (rad * rbe - rbd * rae), accumulate=True)

    return res

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
    dens = torch.zeros((nstat, nstat))
    for i in hole:
        dens[i, i] = 1.0
    return dens


def _HF_iter_ref(
    h1: torch.tensor,
    v2_indices: torch.tensor,
    v2_values: torch.tensor,
    w3_indices: torch.tensor,
    w3_values: torch.tensor,
    # op2: TwoBodyOperator,
    # op3: ThreeBodyOperator,
    npart: int,
    dens: torch.tensor,
    mix: float,
    gamma_buf: torch.tensor,
    omega_buf: torch.tensor,
    hf_ham_buf: torch.tensor,
    dens_buf: torch.tensor
):

    gamma_buf.fill_(0.0)
    omega_buf.fill_(0.0)

    _contract_2nf_kernel(v2_indices, v2_values, dens, gamma_buf)
    _contract_3nf_kernel(w3_indices, w3_values, dens, omega_buf)

    hf_ham_buf.copy_(h1)
    hf_ham_buf += gamma_buf
    hf_ham_buf += 0.5 * omega_buf

    # Compute Energy
    # E_op = h + 0.5 * Gamma + 1/6 * Omega
    # E = Tr( (h + 0.5*Gamma + 1/6*Omega) * rho )
    #   = Tr(h rho) + 0.5*Tr(Gamma rho) + 1/6*Tr(Omega rho)
    e_h1 = torch.sum(h1 * dens)
    e_gamma = torch.sum(gamma_buf * dens)
    e_omega = torch.sum(omega_buf * dens)
    
    erg = e_h1 + 0.5 * e_gamma + (1.0 / 6.0) * e_omega

    # NOTE(vivek): new bottleneck
    vals, vecs = torch.linalg.eigh(hf_ham_buf)
    # print(vals)

    # Select occupied orbitals (first npart columns)
    occ = vecs[:, :npart]
    
    # new_dens = occ @ occ.T.
    # O(N^3) BLAS
    torch.matmul(occ, occ.T, out=dens_buf)


    dens_buf -= dens
    diff_dens = torch.sum(torch.abs(dens_buf))

    if mix != 0:
        dens_buf *= mix
    dens += dens_buf

    return erg, diff_dens, vecs

hf_iter = torch.compile(_HF_iter_ref, dynamic=True, mode='reduce-overhead')

def solve_HF(
    op1: OneBodyOperator,
    op2: TwoBodyOperator,
    op3: ThreeBodyOperator,
    dens: torch.tensor,
    mix: float = 0.5,
    eps: float = 1.0e-8,
    max_iter: int = 100,
    verbose: bool = False,
):
    """
    Solve the Hartree-Fock problem using Zero-Copy strategy.
    """
    converged = False
    torch._dynamo.mark_dynamic(dens, 0)
    torch._dynamo.mark_dynamic(dens, 1)
    torch._dynamo.mark_dynamic(op2.indices, 0)
    torch._dynamo.mark_dynamic(op3.indices, 0)

    nstat = dens.shape[0]
    _dens = dens.clone() 
    gamma_buf = torch.zeros((nstat, nstat), dtype=torch.float64)
    omega_buf = torch.zeros((nstat, nstat), dtype=torch.float64)
    ham_buf = torch.zeros((nstat, nstat), dtype=torch.float64)
    dens_buf = torch.zeros((nstat, nstat), dtype=torch.float64)
    
    h1_dense = op1.to_dense()

    erg0 = 0.0

    op2_indices, op2_values = op2.indices, op2.values
    op3_indices, op3_values = op3.indices, op3.values
    for i in range(max_iter):
        erg, diff_dens, vecs = hf_iter(
            h1_dense, 
            op2_indices, 
            op2_values, 
            op3_indices, 
            op3_values, 
            int(dens.trace().round()),
            _dens, 
            mix,
            gamma_buf,
            omega_buf,
            ham_buf,
            dens_buf
        )
        
        diff = abs(erg - erg0)
        if verbose:
             print(f"Iter {i}: E={erg}, dE={diff}, dRho={diff_dens}")

        if (diff_dens < eps or diff < 1e-12) and i > 1:
            converged = True
            break
            
        erg0 = erg
        
    return erg, vecs, converged
