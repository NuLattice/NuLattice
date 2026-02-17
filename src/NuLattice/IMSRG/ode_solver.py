# Copyright 2025 Matthias Heinz. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Module to solve the IMSRG equations via direct integration of flow equations."""
__authors__ = ["Matthias Heinz"]
__credits__ = ["Matthias Heinz"]
__copyright__ = "(c) Matthias Heinz"
__license__ = "BSD-3-Clause"
__date__ = "2025-09-03"

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent / ".." / ".."))

import numpy as np

from NuLattice.IMSRG import generator
from NuLattice.IMSRG import commutators

import scipy.integrate as integ


def flatten_hamiltonian(dim, e0, f, gamma):
    """
    Flattens Hamiltonian components into a single array for ODE integration

    Packs the scalar energy, one-body operator, and two-body operator into a
    single one-dimensional array suitable for scipy ODE solvers

    :param dim:     Dimension of the single-particle basis
    :type dim:      int
    :param e0:      Zero-body (scalar) energy contribution
    :type e0:       float
    :param f:       One-body operator matrix
    :type f:        numpy array
    :param gamma:   Two-body operator tensor
    :type gamma:    numpy array
    :return:        Flattened array containing all Hamiltonian components
    :rtype:         numpy array
    """
    f_start = 1
    f_end = f_start + dim**2
    gamma_start = f_end
    gamma_end = gamma_start + dim**4

    packed = np.zeros(shape=(gamma_end))
    packed[0] = e0
    packed[f_start:f_end] = np.reshape(f, dim**2)
    packed[gamma_start:gamma_end] = np.reshape(gamma, dim**4)

    return packed


def unflatten_hamiltonian(dim, packed):
    """
    Reconstructs Hamiltonian components from flattened array

    Unpacks a one-dimensional array back into scalar energy, one-body matrix,
    and two-body tensor components for IMSRG calculations

    :param dim:     Dimension of the single-particle basis
    :type dim:      int
    :param packed:  Flattened array containing Hamiltonian components
    :type packed:   numpy array
    :return:        Zero-body energy, one-body matrix, and two-body tensor
    :rtype:         float, numpy array, numpy array
    """
    f_start = 1
    f_end = f_start + dim**2
    gamma_start = f_end
    gamma_end = gamma_start + dim**4

    e0 = packed[0]
    f = np.array(np.reshape(packed[f_start:f_end], (dim, dim)))
    gamma = np.array(np.reshape(packed[gamma_start:gamma_end], (dim, dim, dim, dim)))

    return e0, f, gamma


def write_op_1b(dim, op, fname):
    """
    Writes one-body operator matrix elements to file

    Outputs non-zero matrix elements of a one-body operator to a text file
    in the format: index_p index_q matrix_element

    :param dim:     Dimension of the operator matrix
    :type dim:      int
    :param op:      One-body operator matrix
    :type op:       numpy array
    :param fname:   Output filename
    :type fname:    str
    """
    with open(fname, "w") as fout:
        for p in range(dim):
            for q in range(dim):
                me = op[p, q]
                if abs(me) > 1e-9:
                    fout.write(f"{p:>4} {q:>4} {me:18.10f}\n")


def write_op_2b(dim, op, fname):
    """
    Writes two-body operator matrix elements to file

    Outputs non-zero matrix elements of a two-body operator to a text file
    in the format: index_p index_q index_r index_s matrix_element

    :param dim:     Dimension of each operator index
    :type dim:      int
    :param op:      Two-body operator tensor
    :type op:       numpy array
    :param fname:   Output filename
    :type fname:    str
    """
    with open(fname, "w") as fout:
        for p in range(dim):
            for q in range(dim):
                for r in range(dim):
                    for s in range(dim):
                        me = op[p, q, r, s]
                        if abs(me) > 1e-9:
                            fout.write(f"{p:>4} {q:>4} {r:>4} {s:>4} {me:18.10f}\n")


def norm(x):
    """
    Computes the Frobenius norm of a tensor

    :param x:       Input tensor
    :type x:        numpy array
    :return:        Frobenius norm (sum of squared elements)
    :rtype:         float
    """
    return np.sqrt(np.sum(np.power(x, 2)))


def imsrg_rhs(s, packed, occs, delta, dim, data_tracking, eta_criterion=1e-3):
    """
    Right-hand side function for IMSRG flow equations

    Computes the derivatives dH/ds for the IMSRG flow equations using the
    commutator [eta, H] where eta is the arctan generator. This is not a strictly pure function;
    data_tracking is modified by the function to track integration data

    :param s:               Flow parameter
    :type s:                float
    :param packed:          Flattened Hamiltonian components
    :type packed:           numpy array
    :param occs:            Occupation numbers for reference state
    :type occs:             numpy array
    :param delta:           Energy shift parameter for generator denominators
    :type delta:            float
    :param dim:             Dimension of single-particle basis
    :type dim:              int
    :param data_tracking:   List for storing flow data during integration
    :type data_tracking:    list
    :param eta_criterion:   Optional; Generator norm at which solution should truncate. Default = 1e-3
    :type eta_criterion:    float
    :return:                Flattened derivatives for ODE integration
    :rtype:                 numpy array
    """
    e, f, gamma = unflatten_hamiltonian(dim, packed)

    gen1 = generator.build_1b_arctan_generator(occs, f, delta)
    gen2 = generator.build_2b_arctan_generator(occs, f, gamma, delta)

    norm_gen1 = norm(gen1)
    norm_gen2 = norm(gen2)

    print(
        "s = {:>10.5f}, E = {:>13.5f}, ||gen1|| = {:>15.8f}, ||gen2|| = {:>15.8f}".format(
            s, e, norm_gen1, norm_gen2
        )
    )

    data_tracking.append((s, e, norm_gen1, norm_gen2))

    if np.sqrt(norm_gen1**2 + norm_gen2**2) < eta_criterion:
        return np.zeros_like(packed)

    dh0, dh1, dh2 = commutators.evaluate_imsrg2_commutator(occs, gen1, gen2, f, gamma)

    return flatten_hamiltonian(dim, dh0, dh1, dh2)


def solve_imsrg2(
    occs, e0, f, gamma, s_init=0.0, s_max=40, delta=0.0, eta_criterion=1e-3
):
    """
    Solves IMSRG(2) flow equations in a single integration step

    Integrates the IMSRG flow equations from initial to final flow parameter
    values, returning the converged energy (and flow data for possible further analysis)

    :param occs:            Occupation numbers for reference state
    :type occs:             numpy array
    :param e0:              Initial zero-body energy
    :type e0:               float
    :param f:               Initial one-body operator
    :type f:                numpy array
    :param gamma:           Initial two-body operator
    :type gamma:            numpy array
    :param s_init:          Optional; initial flow parameter value, default = 0.0
    :type s_init:           float
    :param s_max:           Optional; final flow parameter value, default = 40.0
    :type s_max:            float
    :param delta:           Optional; energy shift for generator denominators
    :type delta:            float
    :param eta_criterion:   Optional; Generator norm at which solution should truncate. Default = 1e-3
    :type eta_criterion:    float
    :return:                Final IMSRG energy and flow tracking data
    :rtype:                 float, list[tuple[float, float, float, float]]
    """
    dim = len(occs)

    s = s_init

    data_tracking = []

    solver = integ.ode(imsrg_rhs)
    solver.set_integrator("dopri5", atol=10 ** (-8), rtol=10 ** (-8), nsteps=10 ** (5))
    solver.set_f_params(occs, delta, dim, data_tracking, eta_criterion)
    solver.set_initial_value(
        flatten_hamiltonian(dim, e0, f, gamma),
        s,
    )
    solver.integrate(s_max)
    if solver.successful():
        e0, f, gamma = unflatten_hamiltonian(dim, solver.y)
    else:
        raise Exception("Integration failed.")

    e_imsrg = e0

    data_tracking = [
        x
        for i, x in enumerate(data_tracking)
        if i == len(data_tracking) - 1 or x[0] < data_tracking[i + 1][0]
    ]

    return e_imsrg, data_tracking
