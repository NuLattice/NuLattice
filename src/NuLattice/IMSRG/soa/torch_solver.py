import torch
from typing import Tuple, Callable, Iterable
from . import torch_generator as generator
from . import torch_commutator as commutators

# fuse the 9 kernels together
evaluate_imsrg2_commutator = torch.compile(commutators.evaluate_imsrg2_commutator)

# --- DOPRI5 CONSTANTS (The Butcher Tableau) ---
# Hardcoded for performance
A21 = 0.2
A31 = 3.0 / 40.0
A32 = 9.0 / 40.0
A41 = 44.0 / 45.0
A42 = -56.0 / 15.0
A43 = 32.0 / 9.0
A51 = 19372.0 / 6561.0
A52 = -25360.0 / 2187.0
A53 = 64448.0 / 6561.0
A54 = -212.0 / 729.0
A61 = 9017.0 / 3168.0
A62 = -355.0 / 33.0
A63 = 46732.0 / 5247.0
A64 = 49.0 / 176.0
A65 = -5103.0 / 18656.0
A71 = 35.0 / 384.0
A73 = 500.0 / 1113.0
A74 = 125.0 / 192.0
A75 = -2187.0 / 6784.0
A76 = 11.0 / 84.0

# Error estimate weights (E = result - lower_order_result)
E1 = 35.0 / 384.0 - 5179.0 / 57600.0
E3 = 500.0 / 1113.0 - 7571.0 / 16695.0
E4 = 125.0 / 192.0 - 393.0 / 640.0
E5 = -2187.0 / 6784.0 - -92097.0 / 339200.0
E6 = 11.0 / 84.0 - 187.0 / 2100.0
E7 = -1.0 / 40.0

# NOTE(vivek): Ensure 'e' is a 0-dim Tensor (torch.tensor(0.0)), not a float, for consistency.
State = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def tree_map(func: Callable, *trees: Iterable) -> Tuple:
    """
    Lightweight PyTorch equivalent of jax.tree_map for flat tuples.
    Assumes all trees have the same structure.
    """
    return tuple(func(*args) for args in zip(*trees))


def imsrg_rhs(
    s: float | torch.Tensor,
    state: State,
    occs: torch.Tensor,
    delta: float,
    eta_criterion: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Right-hand side function for IMSRG flow equations

    Computes the derivatives dH/ds for the IMSRG flow equations using the
    commutator [eta, H] where eta is the arctan generator. This is not a strictly pure function;
    data_tracking is modified by the function to track integration data

    :param s:               Flow parameter (float or scalar Tensor)
    :param state:           Tuple of (Energy, Fock, Gamma) Tensors
    :param occs:            Occupation numbers Tensor
    :param delta:           Energy shift parameter
    :param eta_criterion:   Convergence threshold
    :return:                Tuple of derivative Tensors (dE, dF, dGamma)
    """
    e, f, gamma = state

    gen1 = generator.build_1b_arctan_generator(occs, f, delta)
    gen2 = generator.build_2b_arctan_generator(occs, f, gamma, delta)

    norm_gen1 = torch.linalg.norm(gen1)
    norm_gen2 = torch.linalg.norm(gen2)

    total_norm = torch.sqrt(norm_gen1**2 + norm_gen2**2)

    # Commutators [eta, H]
    # Calculate these unconditionally to keep the computation graph static
    dh0, dh1, dh2 = evaluate_imsrg2_commutator(occs, gen1, gen2, f, gamma)

    is_converged = total_norm < eta_criterion

    if not isinstance(dh0, torch.Tensor):
        dh0 = torch.tensor(dh0, dtype=e.dtype, device=e.device)

    final_dh0 = torch.where(
        is_converged, torch.tensor(0.0, device=e.device, dtype=e.dtype), dh0
    )
    final_dh1 = torch.where(is_converged, torch.zeros_like(f), dh1)
    final_dh2 = torch.where(is_converged, torch.zeros_like(gamma), dh2)

    return (final_dh0, final_dh1, final_dh2)


@torch.compile
def dopri5_step(
    rhs_fn: Callable, s: float, dt: float, y: State, args: tuple
) -> Tuple[State, State, State]:
    # --- Stage 1 ---
    k1 = rhs_fn(s, y, *args)

    # --- Stage 2 ---
    y2 = tree_map(lambda y_, k1_: y_ + dt * (A21 * k1_), y, k1)
    k2 = rhs_fn(s + 0.2 * dt, y2, *args)

    # --- Stage 3 ---
    y3 = tree_map(lambda y_, k1_, k2_: y_ + dt * (A31 * k1_ + A32 * k2_), y, k1, k2)
    k3 = rhs_fn(s + 0.3 * dt, y3, *args)

    # --- Stage 4 ---
    y4 = tree_map(
        lambda y_, k1_, k2_, k3_: y_ + dt * (A41 * k1_ + A42 * k2_ + A43 * k3_),
        y,
        k1,
        k2,
        k3,
    )
    k4 = rhs_fn(s + 0.8 * dt, y4, *args)

    # --- Stage 5 ---
    y5 = tree_map(
        lambda y_, k1_, k2_, k3_, k4_: (
            y_ + dt * (A51 * k1_ + A52 * k2_ + A53 * k3_ + A54 * k4_)
        ),
        y,
        k1,
        k2,
        k3,
        k4,
    )
    k5 = rhs_fn(s + (8 / 9) * dt, y5, *args)

    # --- Stage 6 ---
    y6 = tree_map(
        lambda y_, k1_, k2_, k3_, k4_, k5_: (
            y_ + dt * (A61 * k1_ + A62 * k2_ + A63 * k3_ + A64 * k4_ + A65 * k5_)
        ),
        y,
        k1,
        k2,
        k3,
        k4,
        k5,
    )
    k6 = rhs_fn(s + dt, y6, *args)

    # --- Stage 7 ---
    y_next = tree_map(
        lambda y_, k1_, k3_, k4_, k5_, k6_: (
            y_ + dt * (A71 * k1_ + A73 * k3_ + A74 * k4_ + A75 * k5_ + A76 * k6_)
        ),
        y,
        k1,
        k3,
        k4,
        k5,
        k6,
    )

    # k7 is evaluated at y_next
    k7 = rhs_fn(s + dt, y_next, *args)

    # --- Error Estimation ---
    y_err = tree_map(
        lambda k1_, k3_, k4_, k5_, k6_, k7_: (
            dt * (E1 * k1_ + E3 * k3_ + E4 * k4_ + E5 * k5_ + E6 * k6_ + E7 * k7_)
        ),
        k1,
        k3,
        k4,
        k5,
        k6,
        k7,
    )

    return y_next, y_err, k7


def error_ratio(error_tree, atol=1e-6, rtol=1e-6, y=None):
    """
    Calculates the error ratio for adaptive stepping.
    """

    def sq_err(err, y_val):
        scale = atol + torch.abs(y_val) * rtol
        return torch.sum((err / scale) ** 2)

    squared_errors = tree_map(sq_err, error_tree, y)
    sum_sq_error = sum(squared_errors)
    num_elements = sum(x.numel() for x in y)

    return torch.sqrt(sum_sq_error / num_elements)


def solve_imsrg2(
    occs: torch.Tensor,
    e0: float | torch.Tensor,
    f: torch.Tensor,
    gamma: torch.Tensor,
    s_init:float=0.0,
    s_max:float=40.0,
    delta:float=0.0,
    eta_criterion:float=1e-3,
    track_data:bool=True,
):
    """
    Solves IMSRG(2) flow equations using PyTorch-accelerated DOPRI5.

    Integrates the IMSRG flow equations from initial to final flow parameter
    values, returning the converged energy (and flow data for possible further analysis)
    """
    if not isinstance(e0, torch.Tensor):
        e0 = torch.tensor(e0, dtype=f.dtype, device=f.device)

    y = (e0, f, gamma)

    s = s_init
    dt = 0.01

    # do not recompile for variables
    s_t = torch.tensor(s, device=occs.device)
    dt_t = torch.tensor(dt, device=occs.device)
    torch._dynamo.mark_dynamic(s_t, 0)
    torch._dynamo.mark_dynamic(dt_t, 0)

    rhs_args = (occs, delta, eta_criterion)
    data_tracking = []

    rtol, atol, safety = 1e-6, 1e-6, 0.9

    if track_data:
        print(f"{'s':>8}  {'Energy':>14}  {'||eta1||':>12}  {'||eta2||':>12}")
        print("-" * 55)

    while s < s_max:
        if s + dt > s_max:
            dt = s_max - s

        s_t.fill_(s)
        dt_t.fill_(dt)
        y_new, y_err, _ = dopri5_step(imsrg_rhs, s_t, dt_t, y, rhs_args)

        err_val = error_ratio(y_err, atol, rtol, y).item()
        if err_val < 1.0:
            y = y_new
            s += dt

            curr_e = y[0].item()

            gen1 = generator.build_1b_arctan_generator(occs, y[1], delta)
            gen2 = generator.build_2b_arctan_generator(occs, y[1], y[2], delta)
            n1 = torch.linalg.norm(gen1).item()
            n2 = torch.linalg.norm(gen2).item()

            if track_data:
                data_tracking.append((s, curr_e, n1, n2))
                print(f"{s:8.4f}  {curr_e:14.6f}  {n1:12.6f}  {n2:12.6f}")

            if n1**2 + n2**2 < eta_criterion**2:
                if track_data:
                    print(f"Converged at s = {s:.4f}")
                return curr_e, data_tracking

            dt = min(dt * safety * (err_val**-0.2), dt * 5.0)
        else:
            dt = max(dt * safety * (err_val**-0.25), dt * 0.1)

    return y[0].item(), data_tracking
