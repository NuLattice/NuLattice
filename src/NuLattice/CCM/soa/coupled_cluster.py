import torch

from NuLattice._torch_types import TwoBodyOperator, ThreeBodyOperator
import NuLattice.lattice as lat

from . import ccDgrams as dgrams
from . import three_body_utils as tbu

def to_tensor(arr, device=None, dtype=torch.float64):
    """Helper to convert numpy arrays/lists/Operators to Torch tensors."""
    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=dtype)
    if hasattr(arr, "to_dense"):  # Handle OneBodyOperator
        return torch.tensor(arr.to_dense(), device=device, dtype=dtype)
    return torch.tensor(arr, device=device, dtype=dtype)


def to_soa_sparse(sparse_input, device=None, dtype=torch.float64):
    """
    Extracts SoA tensors from a TwoBodyOperator for diagrammatic contractions.
    Assumes a full Torch pipeline where input is already an Operator object.

    :param sparse_input: TwoBodyOperator (indices shape [N, 4], values shape [N])
    :return: (indices, values)
             indices: (4, N) LongTensor formatted for ccDgrams
             values: (N) Tensor
    """
    if isinstance(sparse_input, TwoBodyOperator):
        op = sparse_input.to_torch(device)
    else:
        op = TwoBodyOperator.from_list(
            sparse_input, nstat=0, use_torch=True, device=device
        )

    if len(op) == 0:
        return (
            torch.empty((4, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=dtype, device=device),
        )

    # ccDgrams kernels expect (4, N) indices. Operator stores (N, 4).
    # .T provides a view (no-copy) which is perfect for fused kernels.
    indices = op.indices.T.to(dtype=torch.long)
    values = op.values.to(dtype=dtype)

    return indices, values


def get_fock_matrices(part, hole, myTkin, v_phph, v_phhh, v_hhhh):
    """
    Constructs Fock matrices using torch (setup phase).
    """
    pnum = len(part)
    hnum = len(hole)
    n_states = pnum + hnum
    device = v_phph.device
    dtype = v_phph.dtype

    f_pp = torch.zeros((pnum, pnum), device=device, dtype=dtype)
    f_ph = torch.zeros((pnum, hnum), device=device, dtype=dtype)
    f_hh = torch.zeros((hnum, hnum), device=device, dtype=dtype)

    h_dense = torch.zeros((n_states, n_states), device=device, dtype=dtype)
    p = myTkin.indices[:, 0]
    q = myTkin.indices[:, 1]
    tkin_values = myTkin.values
    h_dense.index_put_((p, q), tkin_values, accumulate=True)

    p_idx = torch.tensor(part, device=device)
    h_idx = torch.tensor(hole, device=device)

    f_pp = h_dense[p_idx[:, None], p_idx]
    f_ph = h_dense[p_idx[:, None], h_idx]
    f_hh = h_dense[h_idx[:, None], h_idx]

    f_pp += torch.einsum("aibi->ab", v_phph)
    f_ph += torch.einsum("aibi->ab", v_phhh)
    f_hh += torch.einsum("aibi->ab", v_hhhh)

    return f_pp, f_ph, f_hh


def get_norm_ord_int(
    thisL, holes, vT1, vS1, str_3NF=0, sparse=True, device=None, dtype=torch.float64
):
    """
    Generates normal-ordered Hamiltonian components using a full Torch SoA pipeline.
    """
    if device is None:
        device = torch.device("cpu")

    lattice = lat.get_lattice(thisL)
    myTkin = lat.Tkin(lattice, thisL)
    mycontact = lat.contacts(vT1, vS1, lattice, thisL)
    hole, part = lat.states2PHSpace(holes, thisL)

    hnum, pnum = len(hole), len(part)
    nstat = hnum + pnum

    raw_2b = list(get_all_interactions(part, hole, mycontact, sparse=sparse))

    for i in range(2, 6):
        raw_2b[i] = torch.as_tensor(raw_2b[i], device=device, dtype=dtype)

    if sparse:
        raw_2b[0] = raw_2b[0].to_torch(device)
        raw_2b[1] = raw_2b[1].to_torch(device)
    else:
        raw_2b[0] = torch.as_tensor(raw_2b[0], device=device, dtype=dtype)
        raw_2b[1] = torch.as_tensor(raw_2b[1], device=device, dtype=dtype)

    fock_mats = list(
        get_fock_matrices(part, hole, myTkin, raw_2b[3], raw_2b[4], raw_2b[5])
    )
    fock_mats = [torch.as_tensor(f, device=device, dtype=dtype) for f in fock_mats]

    if str_3NF != 0:
        my3body = lat.NNNcontact(str_3NF, lattice, thisL)

        w_ops = tbu.get_3NF(part, hole, my3body, device=device)

        # Normal Ordered 1-Body from 3NF
        # w_ops[6]=phh_phh, w_ops[7]=phh_hhh, w_ops[8]=hhh_hhh
        dum_fock = tbu.get_3NF_fock(hnum, pnum, w_ops[6], w_ops[7], w_ops[8])
        for i in range(3):
            fock_mats[i] += dum_fock[i]

        # Normal Ordered 2-Body from 3NF
        # w_ops[2]=pph_pph, w_ops[4]=pph_phh, w_ops[5]=pph_hhh
        dum_two_body = tbu.get_3NF_tbme(
            w_ops[2],
            w_ops[4],
            w_ops[5],
            w_ops[6],
            w_ops[7],
            w_ops[8],
            pnum,
            hnum,
            sparse_pppp=sparse,
            sparse_ppph=sparse,
        )

        # Merge 3NF NO2B contributions into base interactions
        if sparse:

            def merge_soa(op1, op2):
                if len(op2) == 0:
                    return op1
                new_idx = torch.cat([op1.indices, op2.indices], dim=0)
                new_vals = torch.cat([op1.values, op2.values], dim=0)
                return TwoBodyOperator(new_idx, new_vals, nstat)

            raw_2b[0] = merge_soa(raw_2b[0], dum_two_body[0])
            raw_2b[1] = merge_soa(raw_2b[1], dum_two_body[1])
        else:
            raw_2b[0] += dum_two_body[0]
            raw_2b[1] += dum_two_body[1]

        for i in range(2, 6):
            raw_2b[i] += dum_two_body[i]

        # fock_mats[2] is f_hh, raw_2b[5] is v_hhhh, w_ops[8] is w_hhh_hhh
        vacEn = get_ref_energy(fock_mats[2], raw_2b[5], w_ops[8])
    else:
        vacEn = get_ref_energy(fock_mats[2], raw_2b[5], None)

    return vacEn, fock_mats, raw_2b

def get_all_interactions(
    part, hole, mycontact, sparse=False, device=None, dtype=torch.float64
):
    """
    Cleaned SoA interaction sorting with dispatch-based logic to ensure parity.
    """
    pnum, hnum = len(part), len(hole)
    nstat = pnum + hnum
    device = device or torch.device("cpu")

    lookup_h = {idx: i for i, idx in enumerate(hole)}
    lookup_p = {idx: i for i, idx in enumerate(part)}

    if sparse:
        v_pppp_list, v_ppph_list = [], []
    else:
        v_pppp = torch.zeros((pnum, pnum, pnum, pnum), device=device, dtype=dtype)
        v_ppph = torch.zeros((pnum, pnum, pnum, hnum), device=device, dtype=dtype)

    v_pphh = torch.zeros((pnum, pnum, hnum, hnum), device=device, dtype=dtype)
    v_phph = torch.zeros((pnum, hnum, pnum, hnum), device=device, dtype=dtype)
    v_phhh = torch.zeros((pnum, hnum, hnum, hnum), device=device, dtype=dtype)
    v_hhhh = torch.zeros((hnum, hnum, hnum, hnum), device=device, dtype=dtype)

    def get_indices_and_signs(a, b, c, d, sector):
        """Standard antisymmetry permutations for unified dispatch."""
        if sector in [("p", "p", "p", "p"), ("p", "p", "h", "h"), ("h", "h", "h", "h")]:
            return ((a, b, c, d), (b, a, c, d), (a, b, d, c), (b, a, d, c)), (
                1.0,
                -1.0,
                -1.0,
                1.0,
            )
        if sector == ("p", "p", "p", "h"):
            return ((a, b, c, d), (b, a, c, d)), (1.0, -1.0)
        if sector == ("p", "h", "h", "h"):
            return ((a, b, c, d), (a, b, d, c)), (1.0, -1.0)
        if sector == ("p", "h", "p", "h"):
            return ((a, b, c, d),), (1.0,)
        return None, None

    indices = mycontact.indices.detach().cpu().numpy()
    values = mycontact.values.detach().cpu().numpy()
    for position, val in zip(indices, values):
        i1, i2, i3, i4 = position
        k_t = [("h" if i in hole else "p") for i in [i1, i2]]
        b_t = [("h" if i in hole else "p") for i in [i3, i4]]

        s_k, s_b = 1.0, 1.0
        if k_t == ["h", "p"]:
            i1, i2, k_t, s_k = i2, i1, ["p", "h"], -1.0
        if b_t == ["h", "p"]:
            i3, i4, b_t, s_b = i4, i3, ["p", "h"], -1.0

        sector = tuple(k_t + b_t)

        mapped = []
        for i, t in zip([i1, i2, i3, i4], sector):
            mapped.append(lookup_p[i] if t == "p" else lookup_h[i])

        target = {
            ("p", "p", "p", "p"): (v_pppp_list if sparse else v_pppp, True),
            ("p", "p", "p", "h"): (v_ppph_list if sparse else v_ppph, True),
            ("p", "p", "h", "h"): (v_pphh, False),
            ("p", "h", "p", "h"): (v_phph, False),
            ("p", "h", "h", "h"): (v_phhh, False),
            ("h", "h", "h", "h"): (v_hhhh, False),
        }.get(sector)
        if target:
            buf, is_sparse_candidate = target
            perms, signs = get_indices_and_signs(*mapped, sector)
            base_val = float(val) * s_k * s_b

            for p, s in zip(perms, signs):
                term = base_val * s
                if is_sparse_candidate and sparse:
                    buf.append([p[0], p[1], p[2], p[3], term])
                else:
                    buf[p] = term

    if sparse:
        v_pppp = TwoBodyOperator.from_list(
            v_pppp_list, nstat, use_torch=True, device=device
        )
        v_ppph = TwoBodyOperator.from_list(
            v_ppph_list, nstat, use_torch=True, device=device
        )

    return v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh


def ccsd_energy(f_ph, v_pphh, t2, t1):
    """
    Computes CCSD correlation energy using Torch.
    """
    e_1 = torch.einsum("ai,ai->", f_ph, t1)
    e_2 = 0.25 * torch.einsum("abij,abij->", v_pphh, t2)
    e_3 = 0.5 * torch.einsum("abij,ai,bj->", v_pphh, t1, t1)
    return (e_1 + e_2 + e_3).item()


def get_ref_energy(no_1b_hh, no_2b_hhhh, w_hhh_hhh=None):
    """Computes Reference Energy."""
    en = 0.0
    hnum = len(no_1b_hh)
    for i in range(hnum):
        en += no_1b_hh[(i, i)]
        for j in range(hnum):
            en -= 0.5 * no_2b_hhhh[(i, j, i, j)]

    if w_hhh_hhh is not None:
        if isinstance(w_hhh_hhh, ThreeBodyOperator):
            en += tbu.get_3NF_Eref(w_hhh_hhh)
        else:
            for ele in w_hhh_hhh:
                [m, i, j, n, k, l, val] = ele
                if (m, i, j) == (n, k, l):
                    en += val / 6.0
    return en


def t1Init(f_ph, f_pp, f_hh, delta):
    """Initializes T1 using Torch."""
    diag_h = torch.diagonal(f_hh)
    diag_p = -torch.diagonal(f_pp)
    denom = (diag_p.unsqueeze(1) + diag_h.unsqueeze(0)) + delta
    return f_ph / denom


def t2Init(f_pp, f_hh, v_pphh, delta):
    diag_h = torch.diagonal(f_hh)
    diag_p = -torch.diagonal(f_pp)  # This contains -epsilon_a

    denom_hh = diag_h.unsqueeze(0) + diag_h.unsqueeze(1)  # j, i -> ij

    denom_pp = diag_p.unsqueeze(0) + diag_p.unsqueeze(1)  # b, a -> ab

    # Outer sum: -epsilon_a - epsilon_b + epsilon_i + epsilon_j
    denom = (
        denom_pp.unsqueeze(2).unsqueeze(3) + denom_hh.unsqueeze(0).unsqueeze(0) + delta
    )
    return v_pphh / denom


def t1Iter(t1, t2, f_ph, f_pp, f_hh, v_phph, v_phhh, v_pphh, v_ppph, sparse=True):
    H1 = f_ph.clone()
    H1 += dgrams.dgram_akci_ck(v_phph, t1)
    H1 += dgrams.dgram_ck_acik(f_ph, t2)
    H1 += dgrams.dgram_cikl_cakl(v_phhh, t2)
    H1 += dgrams.dgram_cdkl_ck_dali(v_pphh, t1, t2)

    X_hh = -f_hh.clone()
    X_pp = f_pp.clone()

    X_hh += dgrams.dgram_ck_ci(f_ph, t1)
    X_pp += dgrams.dgram_ck_ak(f_ph, t1)
    X_hh += dgrams.dgram_bijk_bj(v_phhh, t1)
    X_hh += dgrams.dgram_cdlk_cdli(v_pphh, t2)
    X_pp += dgrams.dgram_dckl_dakl(v_pphh, t2)
    X_hh += dgrams.dgram_cdlk_cl_di(v_pphh, t1)
    X_pp += dgrams.dgram_cdkl_dk_al(v_pphh, t1)

    if sparse:
        if isinstance(v_ppph, (tuple, list)):
            H1 += v_ppph[0]
            X_pp += v_ppph[1]

        elif isinstance(v_ppph, TwoBodyOperator):
            res = dgrams.v_ppph_dgrams(v_ppph, t1, t2)
            H1 += res[0]
            X_pp += res[1]
    else:
        H1 += -0.5 * torch.einsum("cdak, cdki -> ai", v_ppph, t2)
        X_pp -= torch.einsum("cdak, ck -> ad", v_ppph, t1)

    H1 += torch.einsum("ac, ci -> ai", X_pp, t1)
    H1 += torch.einsum("ki, ak -> ai", X_hh, t1)

    # diag_h = -f_ii, diag_p = f_aa
    # Denom = -(f_aa - f_ii)
    diag_h = torch.diagonal(X_hh)
    diag_p = torch.diagonal(X_pp)
    denom = -(diag_p.unsqueeze(1) + diag_h.unsqueeze(0))

    return t1 + (H1 / denom)


def t2Iter(
    t1,
    t2,
    f_ph,
    f_hh,
    f_pp,
    v_pppp,
    v_phph,
    v_phhh,
    v_pphh,
    v_ppph,
    v_hhhh,
    sparse=True,
):
    H2 = v_pphh.clone()

    H2 += dgrams.dgram_klij_abkl(v_hhhh, t2)
    H2 += dgrams.dgram_bkcj_acik(v_phph, t2)
    H2 += dgrams.dgram_bkij_ak(v_phhh, t1)
    H2 += dgrams.dgram_cdkl_acik_dblj(v_pphh, t2)
    H2 += dgrams.dgram_cdkl_cdij_abkl(v_pphh, t2)
    H2 += dgrams.dgram_klij_ak_bl(v_hhhh, t1)
    H2 += dgrams.dgram_bkci_ak_cj(v_phph, t1)
    H2 += dgrams.dgram_cikl_ck_ablj(v_phhh, t1, t2)
    H2 += dgrams.dgram_cikl_al_bcjk(v_phhh, t1, t2)
    H2 += dgrams.dgram_cjkl_ci_abkl(v_phhh, t1, t2)
    H2 += dgrams.dgram_cjkl_ci_ak_bl(v_phhh, t1)
    H2 += dgrams.dgram_cdkl_ci_dj_abkl(v_pphh, t1, t2)
    H2 += dgrams.dgram_cdkl_ak_bl_cdij(v_pphh, t1, t2)
    H2 += dgrams.dgram_cdkl_ci_bl_adkj(v_pphh, t1, t2)
    H2 += dgrams.dgram_cdkl_ci_ak_dj_bl(v_pphh, t1)

    X_hh = -f_hh.clone()
    X_pp = f_pp.clone()

    X_pp += dgrams.dgram_cdkl_bdkl(v_pphh, t2)
    X_hh += dgrams.dgram_cdkl_cdjl(v_pphh, t2)
    X_pp += dgrams.dgram_ck_bk(f_ph, t1)
    X_hh += dgrams.dgram_ck_cj(f_ph, t1)
    X_hh += dgrams.dgram_cdlk_cl_dj(v_pphh, t1)
    X_pp += dgrams.dgram_cdlk_dk_bl(v_pphh, t1)

    if sparse:
        H2 += dgrams.pIJ(v_ppph[2])
        H2 += dgrams.dgram_da_dbij(v_ppph[3], t2)
        H2 += dgrams.dgram_acik_bcjk(v_ppph[4], t2)
        H2 += dgrams.dgram_bijk_ak1(v_ppph[5], t1)
        H2 += dgrams.dgram_bijk_ak2(v_ppph[6], t1)

        ret1, ret2 = dgrams.v_pppp_dgrams(v_pppp, t1, t2)
        H2 += 0.5 * ret1
        H2 += 0.5 * dgrams.pIJ(ret2)
    else:
        H2 += dgrams.pIJ(torch.einsum("abcj, ci -> abij", v_ppph, t1))
        H2 += -dgrams.pAB(
            torch.einsum("cdak, ck, dbij -> abij", v_ppph, t1, t2)
        )
        H2 += dgrams.pIJ(
            dgrams.pAB(torch.einsum("dcak, di, bcjk -> abij", v_ppph, t1, t2))
        )
        H2 += 0.5 * dgrams.pAB(
            torch.einsum("cdbk, ak, cdij -> abij", v_ppph, t1, t2)
        )
        H2 += 0.5 * dgrams.pIJ(
            dgrams.pAB(
                torch.einsum("cdbk, ci, ak, dj -> abij", v_ppph, t1, t1, t1)
            )
        )
        H2 += 0.5 * torch.einsum("abcd, cdij -> abij", v_pppp, t2)
        H2 += 0.5 * dgrams.pIJ(torch.einsum("abcd, ci, dj -> abij", v_pppp, t1, t1))

    H2 += dgrams.pAB(torch.einsum("bc, acij -> abij", X_pp, t2)) + dgrams.pIJ(
        torch.einsum("kj, abik -> abij", X_hh, t2)
    )

    diag_h = torch.diagonal(X_hh)
    diag_p = torch.diagonal(X_pp)
    denom_hh = diag_h.unsqueeze(0) + diag_h.unsqueeze(1)
    denom_pp = diag_p.unsqueeze(0) + diag_p.unsqueeze(1)
    denom = -(denom_pp.unsqueeze(2).unsqueeze(3) + denom_hh.unsqueeze(0).unsqueeze(0))

    return t2 + (H2 / denom)


def ccsd_solver(
    fock_mats,
    two_body_int,
    t1initial=None,
    eps=1e-8,
    maxSteps=1000,
    max_diis=10,
    delta=0,
    mixing=0.5,
    verbose=False,
    sparse=True,
    ccs=False,
    device=None,
    dtype=torch.float64,
):
    device = device or torch.device("cpu")
    f_pp, f_ph, f_hh = [to_tensor(f, device, dtype) for f in fock_mats]
    v_pppp_in, v_ppph_in, v_pphh, v_phph, v_phhh, v_hhhh = [
        to_tensor(x, device, dtype) if not hasattr(x, "indices") else x
        for x in two_body_int
    ]

    if sparse:
        v_pppp = to_soa_sparse(v_pppp_in, device, dtype)
        v_ppph = to_soa_sparse(v_ppph_in, device, dtype)
    else:
        v_pppp, v_ppph = v_pppp_in, v_ppph_in

    t1 = (
        t1Init(f_ph, f_pp, f_hh, delta)
        if t1initial is None
        else to_tensor(t1initial, device, dtype)
    )
    t2 = (
        torch.zeros_like(v_pphh)
        if (ccs or t1initial is not None)
        else t2Init(f_pp, f_hh, v_pphh, delta)
    )

    if max_diis > 0:
        diis_t1 = [t1.clone()]
        diis_t2 = [t2.clone()]
        diis_errors = []

    prevEnergy = ccsd_energy(f_ph, v_pphh, t2, t1)
    if verbose:
        print(f"Step 0: {prevEnergy}")

    for i in range(maxSteps):
        oldT1, oldT2 = t1.clone(), t2.clone()
        v_ppph_results = dgrams.v_ppph_dgrams(v_ppph, t1, t2) if sparse else v_ppph

        t1_new = t1Iter(
            t1, t2, f_ph, f_pp, f_hh, v_phph, v_phhh, v_pphh, v_ppph_results, sparse
        )
        t1 = torch.lerp(t1_new, t1, mixing)

        if not ccs:
            t2_new = t2Iter(
                oldT1,
                t2,
                f_ph,
                f_hh,
                f_pp,
                v_pppp,
                v_phph,
                v_phhh,
                v_pphh,
                v_ppph_results,
                v_hhhh,
                sparse,
            )
            t2 = (mixing * t2 + (1.0 - mixing) * t2_new).clone()

        energy = ccsd_energy(f_ph, v_pphh, t2, t1)
        if verbose:
            print(f'Step {i + 1}: {energy}', "difference =", abs(energy - prevEnergy) / abs(energy))
        if abs(energy - prevEnergy) / max(1.0, abs(energy)) < eps:
            return energy, t1, t2

        # NOTE(vivek): Pulsed DIIS logic like old version.
        # This is major memory hog, optimizing this will shave lots
        if max_diis > 0:
            diis_t1.append(t1.clone())
            diis_t2.append(t2.clone())
            err_vec = torch.cat([(t1 - oldT1).view(-1), (t2 - oldT2).view(-1)])
            diis_errors.append(err_vec)

            if len(diis_errors) > max_diis:
                diis_t1.pop(0)
                diis_t2.pop(0)
                diis_errors.pop(0)

            if len(diis_errors) == max_diis:
                # Solve Pulay
                # NOTE(vivek): skip stacking E to avoid massive mem alloc
                size = len(diis_errors)
                B = torch.zeros((size, size), device=device, dtype=dtype)
                for i in range(size):
                    for j in range(i, size):
                        # symmetric
                        val = torch.dot(diis_errors[i], diis_errors[j])
                        B[i, j] = val
                        B[j, i] = val
                B = B / (B.abs().max() + 1e-16)
                size = B.shape[0]

                A = -torch.ones((size + 1, size + 1), device=device, dtype=dtype)
                A[:size, :size] = B
                A[size, size] = 0
                rhs = torch.zeros(size + 1, device=device, dtype=dtype)
                rhs[size] = -1

                try:
                    c = torch.linalg.solve(A, rhs)[:size]
                    t1.zero_()
                    if not ccs:
                        t2.zero_()
                    for k in range(size):
                        t1 += c[k] * diis_t1[k + 1]  # Skip initial guess at index 0
                        if not ccs:
                            t2 += c[k] * diis_t2[k + 1]
                except RuntimeError:
                    pass

                diis_t1 = [t1.clone()]
                diis_t2 = [t2.clone()]
                diis_errors = []

        if abs(energy) > 1e10 or torch.isnan(torch.tensor(energy)):
            print("Diverged.")
            break
        prevEnergy = energy

    print("Max iterations reached.")
    return energy, t1, t2



def get_norm_ordered_ham(
    thisL,
    holes,
    myTkin,
    mycontact,
    my3body=None,
    sparse=True,
    NO2B=True,
    device=None,
    dtype=torch.float64,
):
    """
    Setup function migrated to full Torch SoA.
    """
    if device is None:
        device = torch.device("cpu")

    hole, part = lat.states2PHSpace(holes, thisL)
    hnum, pnum = len(hole), len(part)
    nstat = pnum + hnum

    # Assume get_all_interactions returns Operator objects for sparse and Torch tensors for dense
    v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh = get_all_interactions(
        part, hole, mycontact, sparse=sparse, device=device, dtype=dtype
    )

    f_pp, f_ph, f_hh = get_fock_matrices(part, hole, myTkin, v_phph, v_phhh, v_hhhh)

    if my3body is not None:
        # w_ppp_pph, w_ppp_phh, w_pph_pph, w_ppp_hhh, w_pph_phh, w_pph_hhh, w_phh_phh, w_phh_hhh, w_hhh_hhh
        w_res = tbu.get_3NF(part, hole, my3body.to_list(), device=device)

        # 3NF -> 1-Body updates
        dum_fock = tbu.get_3NF_fock(hnum, pnum, w_res[6], w_res[7], w_res[8])
        f_pp += dum_fock[0]
        f_ph += dum_fock[1]
        f_hh += dum_fock[2]

        # 3NF -> 2-Body Normal Ordered (Returns Operators for sparse, tensors for dense)
        dum_2b = tbu.get_3NF_tbme(
            w_res[2],
            w_res[4],
            w_res[5],
            w_res[6],
            w_res[7],
            w_res[8],
            pnum,
            hnum,
            sparse_pppp=sparse,
            sparse_ppph=sparse,
        )

        if sparse:

            def merge_ops(op1, op2):
                if len(op2) == 0:
                    return op1
                if len(op1) == 0:
                    return op2
                new_idx = torch.cat([op1.indices, op2.indices], dim=0)
                new_vals = torch.cat([op1.values, op2.values], dim=0)
                return TwoBodyOperator(new_idx, new_vals, nstat)

            v_pppp = merge_ops(v_pppp, dum_2b[0])
            v_ppph = merge_ops(v_ppph, dum_2b[1])
        else:
            v_pppp += dum_2b[0]
            v_ppph += dum_2b[1]

        v_pphh += dum_2b[2]
        v_phph += dum_2b[3]
        v_phhh += dum_2b[4]
        v_hhhh += dum_2b[5]

        vacEn = get_ref_energy(f_hh, v_hhhh, w_res[8])
    else:
        vacEn = get_ref_energy(f_hh, v_hhhh, None)

    two_body = [v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh]
    fock = [f_pp, f_ph, f_hh]

    NO2B_stuff = vacEn, fock, two_body

    if NO2B or my3body is None:
        return NO2B_stuff
    else:
        return NO2B_stuff, w_res
