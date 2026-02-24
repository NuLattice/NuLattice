from ._constants import HBARC, MASS

# old types
from ._types import (
    LatticeState,
    LatticeSite,
    LatticeSites,
    OneBodyElement,
    TwoBodyElement,
    ThreeBodyElement,
    SingleParticleBasis,
)

# new types
from ._types import OneBodyOperator, TwoBodyOperator, ThreeBodyOperator

import numpy as np
from itertools import combinations
from typing import Tuple


def phys_unit(a_lat: float) -> float:
    """
    returns the energy unit from basic units
    """
    return 0.5 * HBARC**2 / (MASS * a_lat**2)


def _get_sp_basis(myL: int, spin: int = 2, isospin: int = 2) -> np.ndarray:
    lattice_sites = np.mgrid[0:myL, 0:myL, 0:myL, 0:isospin, 0:spin]
    return lattice_sites.reshape(5, -1).T


def get_sp_basis(myL: int, spin: int = 2, isospin: int = 2) -> SingleParticleBasis:
    """
    Builds a 3D lattice for nucleons with spin isospin degrees of freedom

    :param myL: number of lattice sites in each direction
    :type myL:  int
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin:    Optional; number of isospin degrees of freedom
    :type isospin:     int
    :return:    List of integer list [i,j,k,tz,sz] where lattice sites are
                labelled by i, j, k (from 0 to myL-1) in direction 1, 2, 3;
                tz=0, 1 and sz=0,1 correspond to isospin tz-1/2 and spin sz-1/2,
                respectively
    :rtype: list[(int, int, int, int, int)]
    """
    return _get_sp_basis(myL, spin, isospin).tolist()


def state2index(state: LatticeState, myL: int, spin: int = 2, isospin: int = 2) -> int:
    """
    given a state list [i,j,k,tz,sz] this function returns the
    index of that state in the list returned by get_sp_basis

    :param state:   the list [i,j,k,tz,sz]
    :type state:    list[(int, int, int, int, int)]
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int
    :return:    index as an integer
    :rtype: int
    """
    k_stride = isospin * spin
    j_stride = myL * k_stride
    i_stride = myL * j_stride

    return (
        state[0] * i_stride
        + state[1] * j_stride
        + state[2] * k_stride
        + state[3] * spin
        + state[4]
    )

def site2index(site: LatticeSite, myL: int) -> int:
    """
    given a site list [i,j,k] this function returns the index of that state in the list
    returned by get_lattice

    :param site:   the list [i,j,k]
    :type site:    list[(int, int, int)]
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :return:        index as an integer
    :rtype:         int
    """
    i = site[0]
    j = site[1]
    k = site[2]
    index = i * myL**2 + j * myL + k
    return index

def get_lattice(myL: int) -> LatticeSites:
    """
    builds a 3D lattice

    :param myL: number of lattice sites in each direction
    :type myL:  int
    :return:    List of integer lists [i,j,k] of lattice sites are labelled
                by i, j, k (from 0 to myL-1) in direction 1, 2, 3
    :rtype:     list[(int, int, int)]
    """
    lattice_sites = np.mgrid[0:myL, 0:myL, 0:myL]
    return lattice_sites.reshape(3, -1).T.tolist()


def left(site_location: int, myL: int) -> int:
    return (site_location - 1) % myL


def right(site_location: int, myL: int) -> int:
    """
    moves a site to the right in 1D, respecting periodic boundary conditions

    :param site_location:    integer location of the site
    :type site_location:     int
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :return:        index of site one to the right of site with index site
    :rtype:         int
    """
    return (site_location + 1) % myL


def _Tkin(
    lattice_sites: LatticeSites, myL: int, spin: int = 2, isospin: int = 2
) -> OneBodyOperator:
    """
    Generates the Kinetic Energy operator as a OneBodyOperator (SoA).
    """
    k_stride = isospin * spin
    j_stride = myL * k_stride
    i_stride = myL * j_stride

    # Basis information
    basis = _get_sp_basis(myL, spin, isospin)
    nstat = len(basis)
    indices = np.arange(nstat, dtype=np.int64)

    # Diagonal elements (2 * 3 dimensions = 6) -> <p|T|p> = 6.0
    diag_indices = np.column_stack([indices, indices])
    diag_values = np.full(nstat, 6.0, dtype=np.float64)

    # Off-diagonal elements (Hopping)
    # Store lists of arrays to stack later
    all_indices = [diag_indices]
    all_values = [diag_values]

    for dim in range(3):
        neighbor_states = basis.copy()
        # Periodic shift in dimension 'dim'
        neighbor_states[:, dim] = (neighbor_states[:, dim] + 1) % myL

        neighbor_indices = (
            neighbor_states[:, 0] * i_stride
            + neighbor_states[:, 1] * j_stride
            + neighbor_states[:, 2] * k_stride
            + neighbor_states[:, 3] * spin
            + neighbor_states[:, 4]
        )

        # Hop Right: <p | T | p+1> = -1.0
        hop_right_idx = np.column_stack([indices, neighbor_indices])
        hop_right_val = np.full(nstat, -1.0, dtype=np.float64)

        # Hop Left: <p+1 | T | p> = -1.0
        hop_left_idx = np.column_stack([neighbor_indices, indices])
        hop_left_val = np.full(nstat, -1.0, dtype=np.float64)

        all_indices.extend([hop_right_idx, hop_left_idx])
        all_values.extend([hop_right_val, hop_left_val])

    # Combine all parts
    final_indices = np.vstack(all_indices)
    final_values = np.concatenate(all_values)

    return OneBodyOperator(final_indices, final_values, nstat)


def Tkin(
    lattice: LatticeSite, myL: int, spin: int = 2, isospin: int = 2
) -> OneBodyElement:
    """
    Wrapper for backward compatibility. Returns list of [i, j, val].
    """
    op = _Tkin(lattice, myL, spin, isospin)
    return op


def _contacts(
    vT1: float,
    vS1: float,
    lattice: LatticeSites,
    myL: int,
    spin: int = 2,
    isospin: int = 2,
) -> TwoBodyOperator:
    """
    Generates the Contact Interaction operator as a TwoBodyOperator (SoA).
    """
    # Strides for spatial mapping
    k_stride = isospin * spin
    j_stride = myL * k_stride
    i_stride = myL * j_stride

    basis_size = (myL**3) * spin * isospin

    # precompute states
    num_local_states = isospin * spin
    local_indices = np.arange(num_local_states)

    # Create all possible pairs (p, q, r, s)
    # p, q are initial states; r, s are final states
    p, q, r, s = np.meshgrid(
        local_indices, local_indices, local_indices, local_indices, indexing="ij"
    )
    p, q, r, s = p.flatten(), q.flatten(), r.flatten(), s.flatten()

    # Antisymmetry: p < q and r < s
    mask = (p < q) & (r < s)

    tz_p, sz_p = divmod(p, spin)
    tz_q, sz_q = divmod(q, spin)
    tz_r, sz_r = divmod(r, spin)
    tz_s, sz_s = divmod(s, spin)

    # Conservation of Total Tz and Total Sz
    mask &= tz_p + tz_q == tz_r + tz_s
    mask &= sz_p + sz_q == sz_r + sz_s

    p, q, r, s = p[mask], q[mask], r[mask], s[mask]
    tz_p, tz_q = tz_p[mask], tz_q[mask]
    sz_p, sz_q = sz_p[mask], sz_q[mask]

    # Calculate interaction values
    # if tz1 == tz2 -> T=1; if sz1 == sz2 -> S=1; else mixed
    values = np.zeros(len(p))
    t1_mask = tz_p == tz_q
    s1_mask = sz_p == sz_q
    mixed_mask = ~(t1_mask | s1_mask)

    values[t1_mask] = vT1
    values[s1_mask] = vS1

    # mixed channel (Tz=0, Sz=0)
    # indx1 in (indx3, indx4) check:
    is_diag = p == r # p < q and r < s, p must match r or s. 
    values[mixed_mask & is_diag] = (vS1 + vT1) * 0.5
    values[mixed_mask & ~is_diag] = (vS1 - vT1) * 0.5

    spatial_lattice = np.array(lattice)
    offsets = (
        spatial_lattice[:, 0] * i_stride
        + spatial_lattice[:, 1] * j_stride
        + spatial_lattice[:, 2] * k_stride
    )

    # broadcasting: [N_lattice, 1] + [1, N_channels]
    final_p = (offsets[:, None] + p).flatten()
    final_q = (offsets[:, None] + q).flatten()
    final_r = (offsets[:, None] + r).flatten()
    final_s = (offsets[:, None] + s).flatten()
    final_vals = np.tile(values, len(offsets))

    indices = np.column_stack([final_p, final_q, final_r, final_s])

    return TwoBodyOperator(indices, final_vals, basis_size)


def contacts(
    vT1: float,
    vS1: float,
    lattice: LatticeSites,
    myL: int,
    spin: int = 2,
    isospin: int = 2,
) -> TwoBodyElement:
    """
    Wrapper for backward compatibility. Returns list of [p, q, r, s, val].
    """
    op = _contacts(vT1, vS1, lattice, myL, spin, isospin)
    return op


def _NNNcontact(
    v3NF: float, lattice: LatticeSites, myL: int, spin: int = 2, isospin: int = 2
) -> ThreeBodyOperator:
    """
    Generates the 3-Body Contact operator as a ThreeBodyOperator (SoA).
    """
    k_stride = isospin * spin
    j_stride = myL * k_stride
    i_stride = myL * j_stride
    basis_size = (myL**3) * spin * isospin

    num_local_states = isospin * spin
    local_indices = np.arange(num_local_states)

    # Local triples p < q < r
    triples = np.array(list(combinations(local_indices, 3)))

    if len(triples) == 0:
        # Return empty operator
        return ThreeBodyOperator(
            np.empty((0, 6), dtype=np.int64), np.array([]), basis_size
        )

    p, q, r = triples[:, 0], triples[:, 1], triples[:, 2]
    values = np.full(len(p), float(v3NF))

    # Spatial tiling
    spatial_lattice = np.array(lattice)
    offsets = (
        spatial_lattice[:, 0] * i_stride
        + spatial_lattice[:, 1] * j_stride
        + spatial_lattice[:, 2] * k_stride
    )

    final_p = (offsets[:, None] + p).flatten()
    final_q = (offsets[:, None] + q).flatten()
    final_r = (offsets[:, None] + r).flatten()
    final_vals = np.tile(values, len(offsets))

    # Diagonal in 3-body space: <pqr|W|pqr>
    indices = np.column_stack([final_p, final_q, final_r, final_p, final_q, final_r])

    return ThreeBodyOperator(indices, final_vals, basis_size)


def NNNcontact(
    v3NF: float, lattice: LatticeSites, myL: int, spin: int = 2, isospin: int = 2
) -> ThreeBodyElement:
    """
    Wrapper for backward compatibility. Returns list of [p, q, r, s, t, u, val].
    """
    op = _NNNcontact(v3NF, lattice, myL, spin, isospin)
    return op


def to_legacy_p(op:OneBodyOperator):
    res = np.empty((len(op), 3), dtype=object)
    res[:, 0] = op.indices[:, 0]
    res[:, 1] = op.indices[:, 1]
    res[:, 2] = op.values

    return res

def _p(
    lattice_sites: LatticeSites, myL: int, dim: int, spin: int = 2, isospin: int = 2
) -> OneBodyOperator:
    """
    Generates Momentum operator (derivative) as OneBodyOperator.
    dim: 0=x, 1=y, 2=z
    """
    n_local = isospin * spin
    n_spatial = len(lattice_sites)
    nstat = n_spatial * n_local

    k_stride = n_local
    j_stride = myL * k_stride
    i_stride = myL * j_stride
    strides = np.array([i_stride, j_stride, k_stride])

    indices = np.arange(nstat, dtype=np.int64)

    spatial_coords = np.array(lattice_sites)
    neighbor_coords = spatial_coords.copy()

    # Shift in dimension dim
    neighbor_coords[:, dim] = (neighbor_coords[:, dim] + 1) % myL

    # Calculate flattened neighbor indices
    neighbor_spatial_base = np.sum(neighbor_coords * strides, axis=1)
    local_offsets = np.arange(n_local)
    # (N_spatial * N_local) indices
    neighbor_indices = (neighbor_spatial_base[:, None] + local_offsets).flatten()

    # Right hops: <p | -i d | p+1> = -0.5j
    hop_right_idx = np.column_stack([indices, neighbor_indices])
    hop_right_val = np.full(nstat, -0.5j, dtype=np.complex128)  # Note: Complex!

    # Left hops: <p+1 | -i d | p> = 0.5j
    hop_left_idx = np.column_stack([neighbor_indices, indices])
    hop_left_val = np.full(nstat, 0.5j, dtype=np.complex128)

    final_indices = np.vstack([hop_right_idx, hop_left_idx])
    final_values = np.concatenate([hop_right_val, hop_left_val])

    # NOTE(vivek): Operators are typically float, but momentum is complex.
    # The SoA classes currently assume float64 values.
    # If the simulation requires real Hamiltonians, these usually appear in R^2 etc.
    # For now, we cast to complex if the Operator class supports it, or return complex array.
    # Since standard lattice EFT momentum is purely imaginary/Hermitian,
    # we return OneBodyOperator with complex support if updated, or rely on NumPy typing.

    return OneBodyOperator(final_indices, final_values, nstat)


def p_x(
    lattice_sites: LatticeSites, myL: int, spin: int = 2, isospin: int = 2
) -> OneBodyElement:
    op = _p(lattice_sites, myL, 0, spin, isospin)
    return to_legacy_p(op).tolist()


def p_y(
    lattice_sites: LatticeSites, myL: int, spin: int = 2, isospin: int = 2
) -> OneBodyElement:
    op = _p(lattice_sites, myL, 1, spin, isospin)
    return to_legacy_p(op).tolist()


def p_z(
    lattice_sites: LatticeSites, myL: int, spin: int = 2, isospin: int = 2
) -> OneBodyElement:
    op = _p(lattice_sites, myL, 2, spin, isospin)
    return to_legacy_p(op).tolist()


def states2PHSpace(holeList: LatticeState, myL: int) -> Tuple[Tuple[int], Tuple[int]]:
    holes = []
    for h in holeList:
        holes.append(state2index(h, myL))

    parts = tuple(np.delete(np.arange(myL**3 * 4), holes))
    holes = tuple(holes)
    return holes, parts


def makeState(x: int, y: int, z: int, tz: int, sz: int) -> LatticeState:
    return [x, y, z, int(tz + 0.5), int(sz + 0.5)]
