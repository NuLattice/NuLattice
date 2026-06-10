"""
This module provides functions to define the 3D lattice
"""

__authors__ = ["Thomas Papenbrock", "Maxwell Rothman", "Ben Johnson"]
__credits__ = ["Thomas Papenbrock", "Maxwell Rothman", "Ben Johnson"]
__copyright__ = "(c) Thomas Papenbrock and Maxwell Rothman and Ben Johnson"
__license__ = "BSD-3-Clause"
__date__ = "2025-07-26"

import copy
import numpy as np
from itertools import combinations
from typing import Tuple, TypeAlias, List

from .constants import HBARC, MASS

LatticeState: TypeAlias = List[int]
"""
Represents a single-particle state on the 3D lattice.
Format: [i, j, k, tz, sz] where i,j,k are spatial and tz,sz are isospin/spin.
"""

SingleParticleBasis: TypeAlias = List[LatticeState]
"""
A list containing all single-particle states in the basis.
"""

LatticeSite: TypeAlias = List[Tuple[int, int, int]]
"""
Represents a spatial coordinate on the 3D lattice.
Format: (i, j, k)
"""

LatticeSites: TypeAlias = List[LatticeSite]

OneBodyElement: TypeAlias = Tuple[int, int, float]
"""
A single-particle matrix element in sparse format.
Format: (p, q, value) representing <p|O|q>.
"""

TwoBodyElement: TypeAlias = Tuple[int, int, int, int, float]
"""
A two-body matrix element in sparse format.
Format: (p, q, r, s, value) representing <pq|V|rs>.
Typically stored with p < q and r < s.
"""

ThreeBodyElement: TypeAlias = Tuple[int, int, int, int, int, int, float]
"""
A three-body matrix element in sparse format.
Format: (p, q, r, s, t, u, value) representing <pqr|W|stu>.
Typically stored with p < q < r and s < t < u.
"""

def phys_unit(a_lat: float) -> float:
    """
    returns the energy unit from basic units
    """
    return 0.5 * HBARC**2 / (MASS * a_lat**2)


def _get_sp_basis_original(
    myL: int, spin: int = 2, isospin: int = 2
) -> SingleParticleBasis:
    sp_basis = []
    for i in range(myL):
        for j in range(myL):
            for k in range(myL):
                for iso in range(isospin):
                    for sz in range(spin):
                        sp_basis.append([i, j, k, iso, sz])
    return sp_basis


def _get_sp_basis_np(myL: int, spin: int = 2, isospin: int = 2) -> SingleParticleBasis:
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
    return _get_sp_basis_np(myL, spin, isospin).tolist()


def _state2index_original(
    state: LatticeState, myL: int, spin: int = 2, isospin: int = 2
) -> int:
    i = state[0]
    j = state[1]
    k = state[2]
    tz = state[3]
    sz = state[4]
    index = (
        i * myL**2 * isospin * spin
        + j * myL * isospin * spin
        + k * isospin * spin
        + tz * spin
        + sz
    )
    return index


def _state2index_strided(
    state: LatticeState, myL: int, spin: int = 2, isospin: int = 2
) -> int:
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
    return _state2index_strided(state, myL, spin, isospin)


def _get_lattice_original(myL: int) -> LatticeSites:
    lattice = []
    for i in range(myL):
        for j in range(myL):
            for k in range(myL):
                lattice.append([i, j, k])
    return lattice


def _get_lattice_np(myL: int) -> LatticeSites:
    lattice_sites = np.mgrid[0:myL, 0:myL, 0:myL]
    return lattice_sites.reshape(3, -1).T


def get_lattice(myL: int) -> LatticeSites:
    """
    builds a 3D lattice

    :param myL: number of lattice sites in each direction
    :type myL:  int
    :return:    List of integer lists [i,j,k] of lattice sites are labelled
                by i, j, k (from 0 to myL-1) in direction 1, 2, 3
    :rtype:     list[(int, int, int)]
    """
    return _get_lattice_np(myL).tolist()


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


def _right_if(site_location: int, myL: int) -> int:
    if site_location + 1 < myL:
        res = site_location + 1
    else:
        res = 0
    return res


def _left_if(site_location: int, myL: int) -> int:
    if site_location - 1 >= 0:
        res = site_location - 1
    else:
        res = myL - 1
    return res


def _left_modulus(site_location: int, myL: int):
    return (site_location - 1) % myL


def _right_modulus(site_location: int, myL: int):
    return (site_location + 1) % myL


def left(site_location: int, myL: int) -> int:
    """
    moves a site to the left in 1D, respecting periodic boundary conditions

    :param site_location:    integer location of the site
    :type site_location:     int
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :return:        index of site one to the left of site with index site
    :rtype:         int
    """
    return _left_modulus(site_location, myL)


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
    return _right_modulus(site_location, myL)


def _Tkin_original(
    lattice: LatticeSites, myL: int, spin: int = 2, isospin: int = 2
) -> OneBodyElement:
    mat = []
    for site in lattice:
        i = site[0]
        j = site[1]
        k = site[2]
        # diagonal element from each direction
        val = 2.0 * 3
        for tz in range(isospin):
            for sz in range(spin):
                state1 = [i, j, k, tz, sz]
                indx1 = state2index(state1, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx1, val])
        #
        # hop to the right in x
        r = right(i, myL=myL)  # r,j,k
        val = -1.0
        for tz in range(isospin):
            for sz in range(spin):
                state1 = [i, j, k, tz, sz]
                state2 = [r, j, k, tz, sz]
                indx1 = state2index(state1, myL=myL, spin=spin, isospin=isospin)
                indx2 = state2index(state2, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx2, val])
                mat.append([indx2, indx1, val])  # adds a hop-to-the left matrix element
        #
        # hop to the right in y
        r = right(j, myL=myL)  # i,r,k
        val = -1.0
        for tz in range(isospin):
            for sz in range(spin):
                state1 = [i, j, k, tz, sz]
                state2 = [i, r, k, tz, sz]
                indx1 = state2index(state1, myL=myL, spin=spin, isospin=isospin)
                indx2 = state2index(state2, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx2, val])
                mat.append([indx2, indx1, val])  # adds a hop-to-the left matrix element
        #
        # hop to the right in z
        r = right(k, myL=myL)  # i,j,r
        val = -1.0
        for tz in range(isospin):
            for sz in range(spin):
                state1 = [i, j, k, tz, sz]
                state2 = [i, j, r, tz, sz]
                indx1 = state2index(state1, myL=myL, spin=spin, isospin=isospin)
                indx2 = state2index(state2, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx2, val])
                mat.append([indx2, indx1, val])  # adds a hop-to-the left matrix element
    return mat


def _Tkin_np(
    lattice_sites: LatticeSites, myL: int, spin: int = 2, isospin: int = 2
) -> OneBodyElement:
    k_stride = isospin * spin
    j_stride = myL * k_stride
    i_stride = myL * j_stride

    # get all single-particle indices
    # shape: (n_total,)
    basis = _get_sp_basis_np(myL, spin, isospin)
    indices = np.arange(len(basis))

    # Diagonal elements (2 * 3 dimensions = 6)
    # <p|T|p>
    mat = np.column_stack([indices, indices, np.full(len(indices), 6.0)])

    # off-diagonal
    # For each spatial dimension (x, y, z), we find the index of the neighbor
    # NOTE(vivek): The 'neighbor' in a periodic lattice is just a cyclic shift of indices
    # need to be careful to only shift the spatial part, not spin/isospin
    all_hops = [mat]
    for dim in range(3):
        neighbor_states = basis.copy()
        neighbor_states[:, dim] = (neighbor_states[:, dim] + 1) % myL

        neighbor_indices = (
            neighbor_states[:, 0] * i_stride
            + neighbor_states[:, 1] * j_stride
            + neighbor_states[:, 2] * k_stride
            + neighbor_states[:, 3] * spin
            + neighbor_states[:, 4]
        )

        hop_right = np.column_stack(
            [indices, neighbor_indices, np.full(len(indices), -1.0)]
        )
        hop_left = np.column_stack(
            [neighbor_indices, indices, np.full(len(indices), -1.0)]
        )
        all_hops.extend([hop_right, hop_left])

    return np.vstack(all_hops)


# NOTE(vivek): about 25% slower, but not sure if this will change once we switch to gpu (ie list extension expensive)
# reason for slowness likely cpu prefetcher as (3, N, L) requires strided access
def _Tkin_np_flat(
    lattice_sites: LatticeSites, myL: int, spin: int = 2, isospin: int = 2
) -> OneBodyElement:
    n_local = isospin * spin
    n_spatial = len(lattice_sites)
    n_total = n_spatial * n_local

    k_stride = n_local
    j_stride = myL * k_stride
    i_stride = myL * j_stride
    strides = np.array([i_stride, j_stride, k_stride])

    indices = np.arange(n_total)
    diag_mat = np.column_stack([indices, indices, np.full(n_total, 6.0)])

    spatial_coords = np.array(lattice_sites)

    dims = np.arange(3)
    right_coords = np.tile(spatial_coords, (3, 1, 1))

    for d in dims:
        right_coords[d, :, d] = (right_coords[d, :, d] + 1) % myL

    # broadcast spatial indices across local spin/isospin degrees of freedom
    local_offsets = np.arange(n_local)

    # spatial_idx * strides -> (3, N_spatial)
    # add local_offsets -> (3, N_spatial, N_local)
    r_spatial_indices = np.sum(right_coords * strides, axis=2)  # (3, N_spatial)

    # shape: (3, N_spatial, N_local) -> flattened to (3 * N_total)
    r_indices = (r_spatial_indices[:, :, None] + local_offsets).flatten()

    # source indices tiled to match the 3 dimensions and local states
    p_indices = np.tile(indices, 3)

    # Right Hops: [p, r_idx, -1.0]
    # Left Hops:  [r_idx, p, -1.0] (Symmetry of the Laplacian)
    hop_p = np.concatenate([p_indices, r_indices])
    hop_q = np.concatenate([r_indices, p_indices])
    hop_vals = np.full(len(hop_p), -1.0)

    off_diag_mat = np.column_stack([hop_p, hop_q, hop_vals])

    return np.vstack([diag_mat, off_diag_mat])


def Tkin(
    lattice: LatticeSite, myL: int, spin: int = 2, isospin: int = 2
) -> OneBodyElement:
    """
    computes 1-body kinetic energy matrix elements. Really: the negative dimensionless laplacian

    :param lattice: list of lattice sites returned by get_lattice
    :type lattice:  list[(int, int, int)]
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int
    :return:    list of tuples [i, j, value] where i and j are indices in the single-particle
                basis, and value is the value of the matrix element Tij
    :rtype:     list[(int, int, float)]
    """
    res = _Tkin_np(lattice, myL, spin, isospin).tolist()
    return [[int(i), int(j), int(val)] for i, j, val in res]


def contacts(
    vT1: float,
    vS1: float,
    lattice: LatticeSites,
    myL: int,
    spin: int = 2,
    isospin: int = 2,
) -> TwoBodyElement:
    """
    computes matrix elements for 2-body onsite contacts

    :param vT1:     strength of T=1 coupling
    :type vT1:      float
    :param vS1:     strength of S=1 coupling
    :type vS1:      float
    :param lattice: list of lattice sites returned by get_lattice
    :type lattice:  list[(int, int, int)]
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int
    :return:    list of lists [i, j, k, l, value] where i, j and k, l are indices of two particles
                in the single-particle basis, and value is the value of the matrix element <ij||kl>.
                All elements have i<j and k<l
    :rtype:         list[(int, int, int, int, float)]
    """
    res = _contacts_np(vT1, vS1, lattice, myL, spin, isospin)
    return [[int(p), int(q), int(r), int(s), v] for p, q, r, s, v in res]


# NOTE(vivek): looks O(L**3 * (iso * spin)**4)
def _contacts_original(
    vT1: float,
    vS1: float,
    lattice: LatticeSites,
    myL: int,
    spin: int = 2,
    isospin: int = 2,
) -> TwoBodyElement:
    valueT1 = vT1  # isospin triplet strength
    valueS1 = vS1  # spin triplet strength
    matele = []
    for site in lattice:
        for tz1 in range(isospin):
            for sz1 in range(spin):
                stat1 = copy.deepcopy(site)
                stat1.append(tz1)
                stat1.append(sz1)
                indx1 = state2index(stat1, myL=myL, spin=spin, isospin=isospin)
                for tz2 in range(isospin):
                    for sz2 in range(spin):
                        if tz1 == tz2 and sz1 == sz2:  # not asymetric under exchange
                            continue
                        stat2 = copy.deepcopy(site)
                        stat2.append(tz2)
                        stat2.append(sz2)
                        indx2 = state2index(stat2, myL=myL, spin=spin, isospin=isospin)
                        if (
                            indx2 <= indx1
                        ):  # we only keep properly ordered two-body states
                            continue
                        for tz3 in range(isospin):
                            for sz3 in range(spin):
                                stat3 = copy.deepcopy(site)
                                stat3.append(tz3)
                                stat3.append(sz3)
                                indx3 = state2index(
                                    stat3, myL=myL, spin=spin, isospin=isospin
                                )
                                for tz4 in range(isospin):
                                    if tz1 + tz2 != tz3 + tz4:  # Tz is not conserved
                                        continue
                                    for sz4 in range(spin):
                                        if (
                                            sz1 + sz2 != sz3 + sz4
                                        ):  # Sz is not conserved
                                            continue
                                        if (
                                            tz3 == tz4 and sz3 == sz4
                                        ):  # not asymetric under exchange
                                            continue
                                        stat4 = copy.deepcopy(site)
                                        stat4.append(tz4)
                                        stat4.append(sz4)
                                        indx4 = state2index(
                                            stat4, myL=myL, spin=spin, isospin=isospin
                                        )

                                        if (
                                            indx4 <= indx3
                                        ):  # we only keep properly ordered two-body states
                                            continue
                                        if (
                                            tz1 == tz2
                                        ):  # |Tz|=1, T=1, and S=Sz=0 from antisymmetry
                                            # all isospins are equal
                                            matele.append(
                                                [indx1, indx2, indx3, indx4, valueT1]
                                            )
                                        elif (
                                            sz1 == sz2
                                        ):  # |Sz|=1, S=1, and T=Tz=0 from antisymmetry
                                            # all spins are equal
                                            matele.append(
                                                [indx1, indx2, indx3, indx4, valueS1]
                                            )
                                        else:  # Tz=0 and Sz=0
                                            if indx1 in (indx3, indx4):
                                                matele.append(
                                                    [
                                                        indx1,
                                                        indx2,
                                                        indx3,
                                                        indx4,
                                                        (valueS1 + valueT1) * 0.5,
                                                    ]
                                                )
                                            else:
                                                matele.append(
                                                    [
                                                        indx1,
                                                        indx2,
                                                        indx3,
                                                        indx4,
                                                        (valueS1 - valueT1) * 0.5,
                                                    ]
                                                )
    #
    return matele


# NOTE(vivek) O(n4) setup + O(L3)
def _contacts_np(
    vT1: float,
    vS1: float,
    lattice: LatticeSites,
    myL: int,
    spin: int = 2,
    isospin: int = 2,
) -> TwoBodyElement:
    k_stride = isospin * spin
    j_stride = myL * k_stride
    i_stride = myL * j_stride

    # precompute the states
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

    # Extract tz and sz from local indices
    tz_p, sz_p = divmod(p, spin)
    tz_q, sz_q = divmod(q, spin)
    tz_r, sz_r = divmod(r, spin)
    tz_s, sz_s = divmod(s, spin)

    # Conservation of Total Tz and Total Sz
    mask &= tz_p + tz_q == tz_r + tz_s
    mask &= sz_p + sz_q == sz_r + sz_s

    # Apply mask to get valid local interaction channels
    p, q, r, s = p[mask], q[mask], r[mask], s[mask]
    tz_p, tz_q = tz_p[mask], tz_q[mask]
    sz_p, sz_q = sz_p[mask], sz_q[mask]

    values = np.zeros(len(p))

    # if tz1 == tz2 -> T=1; if sz1 == sz2 -> S=1; else mixed
    t1_mask = tz_p == tz_q
    s1_mask = sz_p == sz_q
    mixed_mask = ~(t1_mask | s1_mask)

    values[t1_mask] = vT1
    values[s1_mask] = vS1

    # mixed channel (Tz=0, Sz=0)
    # indx1 in (indx3, indx4) check:
    is_diag = p == r  # p < q and r < s, p must match r or s.
    values[mixed_mask & is_diag] = (vS1 + vT1) * 0.5
    values[mixed_mask & ~is_diag] = (vS1 - vT1) * 0.5

    # Convert lattice to spatial offsets via tiling
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

    return np.column_stack([final_p, final_q, final_r, final_s, final_vals])


def _NNNcontact_original(
    v3NF: float, lattice: LatticeSites, myL: int, spin: int = 2, isospin: int = 2
) -> ThreeBodyElement:
    value = v3NF
    matele = []
    for site in lattice:
        for tz1 in range(isospin):
            for sz1 in range(spin):
                stat1 = copy.deepcopy(site)
                stat1.append(tz1)
                stat1.append(sz1)
                indx1 = state2index(stat1, myL=myL, spin=spin, isospin=isospin)
                for tz2 in range(isospin):
                    for sz2 in range(spin):
                        if tz1 == tz2 and sz1 == sz2:  # not asymetric under exchange
                            continue
                        stat2 = copy.deepcopy(site)
                        stat2.append(tz2)
                        stat2.append(sz2)
                        indx2 = state2index(stat2, myL=myL, spin=spin, isospin=isospin)
                        if indx2 <= indx1:
                            continue
                        for tz3 in range(isospin):
                            for sz3 in range(spin):
                                if (
                                    tz1 == tz3 and sz1 == sz3
                                ):  # not asymetric under exchange
                                    continue
                                if (
                                    tz3 == tz2 and sz3 == sz2
                                ):  # not asymetric under exchange
                                    continue
                                stat3 = copy.deepcopy(site)
                                stat3.append(tz3)
                                stat3.append(sz3)
                                indx3 = state2index(
                                    stat3, myL=myL, spin=spin, isospin=isospin
                                )
                                if indx3 <= indx2:
                                    continue

                                matele.append(
                                    [indx1, indx2, indx3, indx1, indx2, indx3, value]
                                )

    return matele


def _NNNcontact_np(
    v3NF: float, lattice: LatticeSites, myL: int, spin: int = 2, isospin: int = 2
) -> ThreeBodyElement:
    k_stride = isospin * spin
    j_stride = myL * k_stride
    i_stride = myL * j_stride

    num_local_states = isospin * spin
    local_indices = np.arange(num_local_states)

    # NOTE(vivek): only need p < q < r because the original code only stores diagonal
    # matrix elements <pqr|V|pqr> for the unit strength contact.
    # If the logic ever changes to allow off-diagonal triples,
    # --> meshgrid(p, q, r, p', q', r') (memory expensive for large n)
    triples = np.array(list(combinations(local_indices, 3)))

    if len(triples) == 0:
        return []

    p = triples[:, 0]
    q = triples[:, 1]
    r = triples[:, 2]

    # In the original code, the value is v3NF for every valid p < q < r
    values = np.full(len(p), float(v3NF))

    spatial_lattice = np.array(lattice)
    offsets = (
        spatial_lattice[:, 0] * i_stride
        + spatial_lattice[:, 1] * j_stride
        + spatial_lattice[:, 2] * k_stride
    )

    # Broadcasting: [N_lattice, 1] + [1, N_triples]
    final_p = (offsets[:, None] + p).flatten()
    final_q = (offsets[:, None] + q).flatten()
    final_r = (offsets[:, None] + r).flatten()

    final_vals = np.tile(values, len(offsets))

    # 5. Final Stack and Format
    # column_stack: [p, q, r, p, q, r, val]
    return np.column_stack(
        [final_p, final_q, final_r, final_p, final_q, final_r, final_vals]
    )


def NNNcontact(
    v3NF: float, lattice: LatticeSites, myL: int, spin: int = 2, isospin: int = 2
) -> ThreeBodyElement:
    """
    computes matrix elements for three-body onsite contact

    :param v3NF:        strength of the 3 nucleon force
    :type v3NF:         float
    :param lattice:     list of lattice sites returned by get_lattice
    :type lattice:      list[(int, int, int)]
    :param myL:         number of lattice sites in each direction
    :type myL:          int
    :param spin:        Optional; number of spin degrees of freedom
    :type spinL:        int
    :param isospin:     Optional; number of isospin degrees of freedom
    :type isospin:      int
    :return:    list of tuples [i1, i2, i3, j1, j2, j3, value] where i1, i2, i3
                and j1, j2, j3 are indices of three particles in the
                single-particle basis, and value is one (unit strength) for the
                matrix element <i1 i2 i3||j1 j2 j3>.
                All elements have i1<i2<i3 and j1<j2<j3
    :rtype:             list[(int, int, int, int, int, int, float)]
    """
    res = _NNNcontact_np(v3NF, lattice, myL, spin, isospin)
    return [
        [int(p), int(q), int(r), int(s), int(t), int(u), v]
        for p, q, r, s, t, u, v in res
    ]


def _p_x_original(
    lattice: LatticeSites, myL: int, spin: int = 2, isospin: int = 2
) -> OneBodyElement:
    """
    computes matrix elements for 1-body momentum operator p_x. Really: -i times d_x

    :param lattice:     list of lattice sites returned by get_lattice
    :type lattice:      list[(int, int, int)]
    :param myL:         number of lattice sites in each direction
    :type myL:          int
    :param spin:        Optional; number of spin degrees of freedom
    :type spinL:        int
    :param isospin:     Optional; number of isospin degrees of freedom
    :type isospin:      int
    :return:            list of tuples [i, j, value] where i and j are indices in the single-particle
                        basis, and value is the value of the matrix element Tij
    :rtype:             list[(int, int, float)]
    """
    mat = []
    for site in lattice:
        i = site[0]
        j = site[1]
        k = site[2]
        # hop to the right in x
        r = right(i, myL=myL)  # r,j,k
        val = -0.5j
        for tz in range(isospin):
            for sz in range(spin):
                state1 = [i, j, k, tz, sz]
                state2 = [r, j, k, tz, sz]
                indx1 = state2index(state1, myL=myL, spin=spin, isospin=isospin)
                indx2 = state2index(state2, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx2, val])
                mat.append(
                    [indx2, indx1, -val]
                )  # adds a hop-to-the left matrix element
    #
    return mat


def _p_y_original(
    lattice: LatticeSites, myL: int, spin: int = 2, isospin: int = 2
) -> OneBodyElement:
    """
    computes matrix elements for 1-body momentum operator p_y. Really: -i times d_y

    :param lattice:     list of lattice sites returned by get_lattice
    :type lattice:      list[(int, int, int)]
    :param myL:         number of lattice sites in each direction
    :type myL:          int
    :param spin:        Optional; number of spin degrees of freedom
    :type spinL:        int
    :param isospin:     Optional; number of isospin degrees of freedom
    :type isospin:      int
    :return:            list of tuples [i, j, value] where i and j are indices in the single-particle
                        basis, and value is the value of the matrix element Tij
    :rtype:             list[(int, int, float)]
    """
    mat = []
    for site in lattice:
        i = site[0]
        j = site[1]
        k = site[2]
        #
        # hop to the right in y
        r = right(j, myL=myL)  # i,r,k
        val = -0.5j
        for tz in range(isospin):
            for sz in range(spin):
                state1 = [i, j, k, tz, sz]
                state2 = [i, r, k, tz, sz]
                indx1 = state2index(state1, myL=myL, spin=spin, isospin=isospin)
                indx2 = state2index(state2, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx2, val])
                mat.append(
                    [indx2, indx1, -val]
                )  # adds a hop-to-the left matrix element
    #
    return mat


def _p_z_original(
    lattice: LatticeSites, myL: int, spin: int = 2, isospin: int = 2
) -> OneBodyElement:
    """
    computes matrix elements for 1-body momentum operator p_z. Really: -i times d_z

    :param lattice:     list of lattice sites returned by get_lattice
    :type lattice:      list[(int, int, int)]
    :param myL:         number of lattice sites in each direction
    :type myL:          int
    :param spin:        Optional; number of spin degrees of freedom
    :type spinL:        int
    :param isospin:     Optional; number of isospin degrees of freedom
    :type isospin:      int
    :return:            list of tuples [i, j, value] where i and j are indices in the single-particle
                        basis, and value is the value of the matrix element Tij
    :rtype:             list[(int, int, float)]
    """
    mat = []
    for site in lattice:
        i = site[0]
        j = site[1]
        k = site[2]
        #
        # hop to the right in z
        r = right(k, myL=myL)  # i,j,r
        val = -0.5j
        for tz in range(isospin):
            for sz in range(spin):
                state1 = [i, j, k, tz, sz]
                state2 = [i, j, r, tz, sz]
                indx1 = state2index(state1, myL=myL, spin=spin, isospin=isospin)
                indx2 = state2index(state2, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx2, val])
                mat.append(
                    [indx2, indx1, -val]
                )  # adds a hop-to-the left matrix element
    #
    return mat


def _p_np(
    lattice_sites: LatticeSites, myL: int, dim: int, spin: int = 2, isospin: int = 2
) -> OneBodyElement:
    """
    dim: 0 for x, 1 for y, 2 for z.
    """
    n_local = isospin * spin
    n_spatial = len(lattice_sites)
    n_total = n_spatial * n_local

    k_stride = n_local
    j_stride = myL * k_stride
    i_stride = myL * j_stride
    strides = np.array([i_stride, j_stride, k_stride])

    indices = np.arange(n_total)

    spatial_coords = np.array(lattice_sites)
    neighbor_coords = spatial_coords.copy()

    neighbor_coords[:, dim] = (neighbor_coords[:, dim] + 1) % myL

    # Convert shifted coordinates back to flat indices
    neighbor_spatial_base = np.sum(neighbor_coords * strides, axis=1)
    local_offsets = np.arange(n_local)
    neighbor_indices = (neighbor_spatial_base[:, None] + local_offsets).flatten()

    res = np.empty((n_total * 2, 3), dtype=object)

    # Right hops
    res[:n_total, 0] = indices
    res[:n_total, 1] = neighbor_indices
    res[:n_total, 2] = -0.5j

    # Left hops
    res[n_total:, 0] = neighbor_indices
    res[n_total:, 1] = indices
    res[n_total:, 2] = 0.5j

    return res


def p_x(
    lattice_sites: LatticeSites, myL: int, spin: int = 2, isospin: int = 2
) -> OneBodyElement:
    return _p_np(lattice_sites, myL, 0, spin, isospin).tolist()


def p_y(
    lattice_sites: LatticeSites, myL: int, spin: int = 2, isospin: int = 2
) -> OneBodyElement:
    return _p_np(lattice_sites, myL, 1, spin, isospin).tolist()


def p_z(
    lattice_sites: LatticeSites, myL: int, spin: int = 2, isospin: int = 2
) -> OneBodyElement:
    return _p_np(lattice_sites, myL, 2, spin, isospin).tolist()


def states2PHSpace(holeList: LatticeState, myL: int) -> Tuple[Tuple[int], Tuple[int]]:
    """
    Takes a list of hole states and returns the hole and particle spaces

    :param holeList:    list of holes and their states as [i, j, k, tz, sz]
    :type holeList:     list[(int, int, int, int, int)]
    :param myL:         number of lattice sites in each direction
    :type myL:          int
    :return:            hole and particle space
    :rtype:             tuple(int), tuple(int)
    """
    holes = []
    for h in holeList:
        holes.append(state2index(h, myL))

    parts = tuple(np.delete(np.arange(myL**3 * 4), holes))
    holes = tuple(holes)
    return holes, parts


def makeState(x: int, y: int, z: int, tz: int, sz: int) -> LatticeState:
    """
    Takes position in x, y, and z on the lattice as well as the spin and isospin and returns a state

    :param x:   x position in lattice
    :type x:    int
    :param y:   y position in lattice
    :type y:    int
    :param z:   z position in lattice
    :type z:    int
    :param tz:  isospin
    :type tz:   0.5 | -0.5
    :param sz:  spin
    :type sz:   0.5 | -0.5
    :return:    a particle state on the lattice as a list
    :rtype:     list[(int, int, int, int, int)]
    """
    return [x, y, z, int(tz + 0.5), int(sz + 0.5)]
