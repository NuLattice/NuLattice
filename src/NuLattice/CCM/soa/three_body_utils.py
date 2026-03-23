import torch
from typing import List, Tuple, Union
from NuLattice._torch_types import ThreeBodyOperator, TwoBodyOperator

ThreeBodyList = List[List[Union[int, float]]]


def get_3NF(
    part: List[int],
    hole: List[int],
    my3body: ThreeBodyList,
    device: torch.device = None,
) -> Tuple[ThreeBodyOperator, ...]:
    """
    Sorts raw three-body matrix elements into 9 blocks (ppp_pph, etc.) using vectorized
    PyTorch operations.

    :param part:    List of particle-space indices
    :param hole:    List of hole-space indices
    :param my3body: List of 3-body matrix elements [i1...i6, val]
    :param device:  Torch device to perform sorting on (default: CPU)
    :return:        9 ThreeBodyOperators containing Torch tensors
    """
    nstat = len(part) + len(hole)

    if not my3body:
        empty_idx = torch.empty((0, 6), dtype=torch.long, device=device)
        empty_val = torch.empty((0,), dtype=torch.float64, device=device)
        return tuple(ThreeBodyOperator(empty_idx, empty_val, nstat) for _ in range(9))

    data_tensor = torch.tensor(my3body, dtype=torch.float64)
    if device is not None:
        data_tensor = data_tensor.to(device)

    indices = data_tensor[:, :6].long()  # (N, 6)
    values = data_tensor[:, 6]  # (N,)

    max_idx = int(torch.max(indices).item())

    # 0=hole, 1=particle, -1=invalid
    type_map = torch.full((max_idx + 1,), -1, dtype=torch.long, device=device)
    local_map = torch.full((max_idx + 1,), -1, dtype=torch.long, device=device)

    h_tens = torch.tensor(hole, device=device, dtype=torch.long)
    p_tens = torch.tensor(part, device=device, dtype=torch.long)

    type_map[h_tens] = 0
    type_map[p_tens] = 1

    local_map[h_tens] = torch.arange(len(hole), device=device)
    local_map[p_tens] = torch.arange(len(part), device=device)

    types = type_map[indices]  # (N, 6)

    ket_types = types[:, :3]
    bra_types = types[:, 3:]

    ket_indices = indices[:, :3]
    bra_indices = indices[:, 3:]

    def vectorize_order(current_types, current_indices):
        """Reorders indices to (p...p h...h) format and computes sign flips."""
        # 3=ppp, 2=pph, 1=phh, 0=hhh
        sums = torch.sum(current_types, dim=1)
        new_indices = current_indices.clone()
        signs = torch.ones(current_indices.shape[0], dtype=torch.float64, device=device)

        # Case: pph (sum=2)
        mask_2 = sums == 2
        if mask_2.any():
            sub_types = current_types[mask_2]
            sub_idx = current_indices[mask_2]

            # php (1,0,1) -> Swap col 1 & 2 -> (1,1,0) pph. Sign -1.
            is_php = sub_types[:, 1] == 0
            if is_php.any():
                idx_php = sub_idx[is_php]
                # Swap indices 1 and 2 (columns)
                new_php = torch.stack(
                    [idx_php[:, 0], idx_php[:, 2], idx_php[:, 1]], dim=1
                )

                global_mask = mask_2.clone()
                global_mask[mask_2] = is_php
                new_indices[global_mask] = new_php
                signs[global_mask] = -1.0

            # hpp (0,1,1) -> Cyclic shift -> (1,1,0) pph. Sign -1.
            is_hpp = sub_types[:, 0] == 0
            if is_hpp.any():
                idx_hpp = sub_idx[is_hpp]
                # hpp -> pph: indices [1, 2, 0]
                new_hpp = torch.stack(
                    [idx_hpp[:, 1], idx_hpp[:, 2], idx_hpp[:, 0]], dim=1
                )

                global_mask = mask_2.clone()
                global_mask[mask_2] = is_hpp
                new_indices[global_mask] = new_hpp
                signs[global_mask] = -1.0

        # Case: phh (sum=1)
        mask_1 = sums == 1
        if mask_1.any():
            sub_types = current_types[mask_1]
            sub_idx = current_indices[mask_1]

            # hph (0,1,0) -> Swap 0 & 1 -> (1,0,0) phh. Sign -1.
            is_hph = sub_types[:, 1] == 1
            if is_hph.any():
                idx_hph = sub_idx[is_hph]
                new_hph = torch.stack(
                    [idx_hph[:, 1], idx_hph[:, 0], idx_hph[:, 2]], dim=1
                )

                global_mask = mask_1.clone()
                global_mask[mask_1] = is_hph
                new_indices[global_mask] = new_hph
                signs[global_mask] = -1.0

            # hhp (0,0,1) -> Cyclic shift -> (1,0,0) phh. Sign -1.
            is_hhp = sub_types[:, 2] == 1
            if is_hhp.any():
                idx_hhp = sub_idx[is_hhp]
                new_hhp = torch.stack(
                    [idx_hhp[:, 2], idx_hhp[:, 0], idx_hhp[:, 1]], dim=1
                )

                global_mask = mask_1.clone()
                global_mask[mask_1] = is_hhp
                new_indices[global_mask] = new_hhp
                signs[global_mask] = -1.0

        return new_indices, signs, sums

    ket_canon, ket_signs, ket_sums = vectorize_order(ket_types, ket_indices)
    bra_canon, bra_signs, bra_sums = vectorize_order(bra_types, bra_indices)

    # (sum_ket, sum_bra)
    bucket_defs = [
        (3, 2),  # ppp_pph
        (3, 1),  # ppp_phh
        (2, 2),  # pph_pph
        (3, 0),  # ppp_hhh
        (2, 1),  # pph_phh
        (2, 0),  # pph_hhh
        (1, 1),  # phh_phh
        (1, 0),  # phh_hhh
        (0, 0),  # hhh_hhh
    ]

    perms_lookup = {
        3: (
            torch.tensor(
                [[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1], [1, 2, 0], [2, 0, 1]],
                device=device,
            ),
            torch.tensor([1, -1, -1, -1, 1, 1], device=device, dtype=torch.float64),
        ),
        0: (
            torch.tensor(
                [[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1], [1, 2, 0], [2, 0, 1]],
                device=device,
            ),
            torch.tensor([1, -1, -1, -1, 1, 1], device=device, dtype=torch.float64),
        ),
        2: (
            torch.tensor([[0, 1, 2], [1, 0, 2]], device=device),
            torch.tensor([1, -1], device=device, dtype=torch.float64),
        ),
        1: (
            torch.tensor([[0, 1, 2], [0, 2, 1]], device=device),
            torch.tensor([1, -1], device=device, dtype=torch.float64),
        ),
    }

    results = []

    for k_s, b_s in bucket_defs:
        mask = (ket_sums == k_s) & (bra_sums == b_s)

        if not mask.any():
            empty_idx = torch.empty((0, 6), dtype=torch.long, device=device)
            empty_val = torch.empty((0,), dtype=torch.float64, device=device)
            results.append(ThreeBodyOperator(empty_idx, empty_val, nstat))
            continue

        base_ket = ket_canon[mask]  # (M, 3)
        base_bra = bra_canon[mask]  # (M, 3)
        base_vals = values[mask] * ket_signs[mask] * bra_signs[mask]  # (M,)

        kp, ks = perms_lookup[k_s]
        bp, bs = perms_lookup[b_s]

        n_perms_k = len(ks)
        n_perms_b = len(bs)

        # (M, 3) -> (M, n_pk, 3)
        # base_ket[:, kp] works if kp is (n_pk, 3)
        expanded_ket = base_ket[:, kp]

        # (M, 3) -> (M, n_pb, 3)
        expanded_bra = base_bra[:, bp]

        # (n_pk, 1) * (1, n_pb) -> (n_pk, n_pb)
        comb_signs = ks.view(-1, 1) * bs.view(1, -1)

        # (M, 1, 1) * (1, n_pk, n_pb) -> (M, n_pk, n_pb)
        expanded_vals = base_vals.view(-1, 1, 1) * comb_signs.view(
            1, n_perms_k, n_perms_b
        )

        # Final shape needs to be (Total, 3) for kets/bras

        # (M, n_pk, 1, 3) -> (M, n_pk, n_pb, 3)
        final_ket = (
            expanded_ket.unsqueeze(2).expand(-1, -1, n_perms_b, -1).reshape(-1, 3)
        )

        # (M, 1, n_pb, 3) -> (M, n_pk, n_pb, 3)
        final_bra = (
            expanded_bra.unsqueeze(1).expand(-1, n_perms_k, -1, -1).reshape(-1, 3)
        )

        final_vals = expanded_vals.reshape(-1)

        local_ket = local_map[final_ket]
        local_bra = local_map[final_bra]

        final_indices = torch.cat([local_ket, local_bra], dim=1)

        results.append(ThreeBodyOperator(final_indices, final_vals, nstat))

    return tuple(results)


def get_3NF_Eref(w_hhh_hhh: ThreeBodyOperator) -> float:
    """Computes Reference Energy contribution from SoA inputs using Torch."""
    indices = w_hhh_hhh.indices
    values = w_hhh_hhh.values

    if len(values) == 0:
        return 0.0

    # Check if m==n (0==3) AND i==k (1==4) AND j==l (2==5)
    mask = (
        (indices[:, 0] == indices[:, 3])
        & (indices[:, 1] == indices[:, 4])
        & (indices[:, 2] == indices[:, 5])
    )

    return torch.sum(values[mask]).item() / 6.0


def get_3NF_fock(
    hnum: int,
    pnum: int,
    w_phh_phh: ThreeBodyOperator,
    w_phh_hhh: ThreeBodyOperator,
    w_hhh_hhh: ThreeBodyOperator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes normal-ordered 1-body contributions (Fock Matrix updates) using Torch.
    """
    device = w_phh_phh.values.device if len(w_phh_phh) > 0 else torch.device("cpu")

    f_pp = torch.zeros((pnum, pnum), dtype=torch.float64, device=device)
    f_ph = torch.zeros((pnum, hnum), dtype=torch.float64, device=device)
    f_hh = torch.zeros((hnum, hnum), dtype=torch.float64, device=device)

    def fock_accumulator(target, op: ThreeBodyOperator, idx_map):
        indices = op.indices
        values = op.values
        if len(values) == 0:
            return

        # Contract over i=k (cols 1,4) and j=l (cols 2,5)
        mask = (indices[:, 1] == indices[:, 4]) & (indices[:, 2] == indices[:, 5])

        valid_rows = indices[mask]
        valid_vals = values[mask]

        if len(valid_vals) == 0:
            return

        row_idx = valid_rows[:, idx_map[0]]
        col_idx = valid_rows[:, idx_map[1]]

        flat_indices = row_idx * target.size(1) + col_idx

        target.view(-1).index_add_(0, flat_indices, 0.5 * valid_vals)

    # F_pp from W_phh_phh [a, i, j, b, k, l] -> a=0, b=3
    fock_accumulator(f_pp, w_phh_phh, (0, 3))

    # F_ph from W_phh_hhh [a, i, j, m, k, l] -> a=0, m=3
    fock_accumulator(f_ph, w_phh_hhh, (0, 3))

    # F_hh from W_hhh_hhh [n, i, j, m, k, l] -> n=0, m=3
    fock_accumulator(f_hh, w_hhh_hhh, (0, 3))

    return f_pp, f_ph, f_hh


def get_3NF_tbme(
    w_pph_pph: ThreeBodyOperator,
    w_pph_phh: ThreeBodyOperator,
    w_pph_hhh: ThreeBodyOperator,
    w_phh_phh: ThreeBodyOperator,
    w_phh_hhh: ThreeBodyOperator,
    w_hhh_hhh: ThreeBodyOperator,
    pnum: int,
    hnum: int,
    sparse_pppp: bool = True,
    sparse_ppph: bool = True,
) -> Tuple[Union[TwoBodyOperator, torch.Tensor], ...]:
    """
    Computes normal-ordered 2-body contributions using Torch.
    Returns sparse (TwoBodyOperator) or dense (torch.Tensor) arrays.
    """
    nstat = pnum + hnum
    device = w_pph_pph.values.device if len(w_pph_pph) > 0 else torch.device("cpu")

    v_pphh = torch.zeros((pnum, pnum, hnum, hnum), dtype=torch.float64, device=device)
    v_phph = torch.zeros((pnum, hnum, pnum, hnum), dtype=torch.float64, device=device)
    v_phhh = torch.zeros((pnum, hnum, hnum, hnum), dtype=torch.float64, device=device)
    v_hhhh = torch.zeros((hnum, hnum, hnum, hnum), dtype=torch.float64, device=device)

    def get_valid_contraction(op: ThreeBodyOperator):
        indices = op.indices
        values = op.values
        if len(values) == 0:
            return torch.empty((0, 6), dtype=torch.long, device=device), torch.empty(
                (0,), dtype=torch.float64, device=device
            )

        # Contract over 3rd and 6th indices
        # [a, b, i, c, d, j] -> i=j
        mask = indices[:, 2] == indices[:, 5]
        return indices[mask], values[mask]

    def add_dense(target, op: ThreeBodyOperator, dim_map):
        valid_idx, valid_val = get_valid_contraction(op)
        if len(valid_val) == 0:
            return

        # Compute flat indices for 4D tensor
        # dim_map maps from 3-body index positions to 2-body tensor dimensions
        # target shape is 4D, so we need strides
        strides = torch.tensor(target.stride(), device=device)

        # Gather relevant indices
        # (N_valid, 4)
        dims_to_keep = valid_idx[:, list(dim_map)]

        flat_indices = torch.sum(dims_to_keep * strides, dim=1)
        target.view(-1).index_add_(0, flat_indices, valid_val)

    def get_sparse(op: ThreeBodyOperator, dim_map) -> TwoBodyOperator:
        valid_idx, valid_val = get_valid_contraction(op)
        if len(valid_val) == 0:
            empty_idx = torch.empty((0, 4), dtype=torch.long, device=device)
            empty_val = torch.empty((0,), dtype=torch.float64, device=device)
            return TwoBodyOperator(empty_idx, empty_val, nstat)

        new_indices = valid_idx[:, list(dim_map)]
        return TwoBodyOperator(new_indices, valid_val, nstat)

    # V_pppp from W_pph_pph [a, b, i, c, d, j] -> i=j -> [a, b, c, d] = [0, 1, 3, 4]
    if sparse_pppp:
        v_pppp = get_sparse(w_pph_pph, (0, 1, 3, 4))
    else:
        v_pppp = torch.zeros(
            (pnum, pnum, pnum, pnum), dtype=torch.float64, device=device
        )
        add_dense(v_pppp, w_pph_pph, (0, 1, 3, 4))

    # V_ppph from W_pph_phh [a, b, i, c, k, j] -> i=j -> [a, b, c, k] = [0, 1, 3, 4]
    if sparse_ppph:
        v_ppph = get_sparse(w_pph_phh, (0, 1, 3, 4))
    else:
        v_ppph = torch.zeros(
            (pnum, pnum, pnum, hnum), dtype=torch.float64, device=device
        )
        add_dense(v_ppph, w_pph_phh, (0, 1, 3, 4))

    # V_pphh from W_pph_hhh [a, b, i, m, k, j] -> [0, 1, 3, 4]
    add_dense(v_pphh, w_pph_hhh, (0, 1, 3, 4))

    # V_phph from W_phh_phh [a, n, i, b, k, j] -> [0, 1, 3, 4]
    add_dense(v_phph, w_phh_phh, (0, 1, 3, 4))

    # V_phhh from W_phh_hhh [a, n, i, l, k, j] -> [0, 1, 3, 4]
    add_dense(v_phhh, w_phh_hhh, (0, 1, 3, 4))

    # V_hhhh from W_hhhh_hhh [m, n, i, l, k, j] -> [0, 1, 3, 4]
    add_dense(v_hhhh, w_hhh_hhh, (0, 1, 3, 4))

    return v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh
