import torch

def antisymmetrize_2b_pq_torch(a2: torch.Tensor) -> torch.Tensor:
    """Antisymmetrizes w.r.t. first two indices (pq)."""
    return 0.5 * (a2 - a2.transpose(0, 1))

def antisymmetrize_2b_rs_torch(a2: torch.Tensor) -> torch.Tensor:
    """Antisymmetrizes w.r.t. last two indices (rs)."""
    return 0.5 * (a2 - a2.transpose(2, 3))

def antisymmetrize_2b_torch(a2: torch.Tensor) -> torch.Tensor:
    """Fully antisymmetrizes a two-body operator."""
    return antisymmetrize_2b_rs_torch(antisymmetrize_2b_pq_torch(a2))

def evaluate_comm_110(occs, a1, b1):
    """[1,1]->0 commutator."""
    occsbar = 1.0 - occs
    # trace(n * nbar * a * b.T) pattern
    weight = occs[:, None] * occsbar[None, :]
    val = torch.sum(weight * a1 * b1.T) - torch.sum(weight.T * a1 * b1.T)
    return val

def evaluate_comm_111(occs, a1, b1):
    """[1,1]->1 commutator: Standard Matrix Commutator [A, B]."""
    return torch.matmul(a1, b1) - torch.matmul(b1, a1)

def evaluate_comm_121(occs, a1, b2):
    """[1,2]->1 commutator."""
    ph_factor = occs[:, None] - occs[None, :]
    return torch.einsum("pq,iqjp->ij", ph_factor * a1, b2)

def evaluate_comm_122(occs, a1, b2):
    """[1,2]->2 commutator."""
    dim = a1.shape[0]
    # ip, pjkl -> ijkl (Batch matmul approach)
    term1 = torch.matmul(a1, b2.reshape(dim, -1)).reshape(dim, dim, dim, dim)
    term2 = torch.einsum("pk,ijpl->ijkl", a1, b2)
    return antisymmetrize_2b_torch(2.0 * (term1 - term2))

def evaluate_comm_220(occs, a2, b2):
    """[2,2]->0 commutator."""
    occsbar = 1.0 - occs
    # Weight tensor: n_p n_q nbar_r nbar_s
    w = (occs[:, None, None, None] * occs[None, :, None, None] * occsbar[None, None, :, None] * occsbar[None, None, None, :])
    
    val = torch.sum(w * a2 * b2.permute(2, 3, 0, 1))
    val -= torch.sum(w.permute(2, 3, 0, 1) * a2 * b2.permute(2, 3, 0, 1))
    return 0.25 * val

def evaluate_comm_221(occs, a2, b2):
    """[2,2]->1 commutator optimized for M4 AMX."""
    dim = len(occs)
    occsbar = 1.0 - occs
    
    # Weight Tensor: (nbar_p nbar_q n_r) + (n_p n_q nbar_r)
    w = (occsbar[None, None, :, None] * occsbar[None, None, None, :] * occs[None, :, None, None]) + \
        (occs[None, None, :, None] * occs[None, None, None, :] * occsbar[None, :, None, None])
    
    a_w = (a2 * w).reshape(dim, -1)
    b_w = (b2 * w).reshape(dim, -1)
    
    # pqjr -> rpqj
    a_tr = a2.permute(3, 0, 1, 2).reshape(-1, dim)
    b_tr = b2.permute(3, 0, 1, 2).reshape(-1, dim)
    
    return 0.5 * (torch.matmul(a_w, b_tr) - torch.matmul(b_w, a_tr))

def evaluate_comm_222_pphh(occs, a2, b2):
    """[2,2]->2 pphh (O(N^6) bottleneck)."""
    dim = len(occs)
    occsbar = 1.0 - occs
    q_factor = (occsbar[:, None] * occsbar[None, :]) - (occs[:, None] * occs[None, :])
    
    # Reshape 4D to 2D matrices for hardware GEMM
    A_mat = a2.reshape(dim**2, dim**2)
    B_mat = b2.reshape(dim**2, dim**2)
    
    # Apply weights across the middle indices
    A_q = (a2 * q_factor[None, None, :, :]).reshape(dim**2, dim**2)
    B_q = (b2 * q_factor[None, None, :, :]).reshape(dim**2, dim**2)
    
    res = 0.5 * (torch.matmul(A_q, B_mat) - torch.matmul(B_q, A_mat))
    return res.reshape(dim, dim, dim, dim)

def evaluate_comm_222_ph(occs, a2, b2):
    """[2,2]->2 ph (O(N^6) bottleneck)."""
    dim = len(occs)
    ph_factor = (occs[:, None] * (1.0 - occs[None, :])) - ((1.0 - occs[:, None]) * occs[None, :])
    
    # Reorder for contraction (pjkq, iqpl -> ijkl)
    # Aligning indices for Matrix Multiplication
    a_ph = (a2 * ph_factor[:, None, None, :]).permute(1, 2, 0, 3).reshape(dim**2, dim**2)
    b_ph = b2.permute(2, 1, 0, 3).reshape(dim**2, dim**2)
    
    c_ph = torch.matmul(a_ph, b_ph).reshape(dim, dim, dim, dim).permute(2, 0, 1, 3)
    return antisymmetrize_2b_torch(-4.0 * c_ph)

def evaluate_imsrg2_commutator(occs, a1, a2, b1, b2):
    res0 = evaluate_comm_110(occs, a1, b1) + evaluate_comm_220(occs, a2, b2)
    
    res1 = (
        evaluate_comm_111(occs, a1, b1)
        + evaluate_comm_121(occs, a1, b2)
        - evaluate_comm_121(occs, b1, a2)
        + evaluate_comm_221(occs, a2, b2)
    )
    
    res2 = (
        evaluate_comm_122(occs, a1, b2)
        - evaluate_comm_122(occs, b1, a2)
        + evaluate_comm_222_pphh(occs, a2, b2) 
        + evaluate_comm_222_ph(occs, a2, b2)
    )

    return res0, res1, res2
