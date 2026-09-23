import torch
import triton
import triton.language as tl


@triton.jit
def compute_L_exp_cumsum_kernel(
    A_ptr, L_ptr,
    Bsz, Csz,
    stride_A_b, stride_A_h, stride_A_c, stride_A_s,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    S: tl.constexpr,  # chunk_size (128)
    H: tl.constexpr,  # num_heads (32)
    num_warps=8, num_stages=2
):
    """
    Compute L[b, c, i, j, h] = exp(sum_{t<=j} A[b, h, c, t]) for i >= j, else 0.
    Grid: (Bsz, Csz). For each (b, c), loop over i and j, maintain cumsum vector s[j].
    """
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Loop over i
    for i in range(S):
        # Vector of s[j] for j in [0..S-1]
        s = tl.zeros((S,), dtype=tl.float32)
        # Loop over j
        for j in range(S):
            # Cumsum across j: s[j] += A[b, h, c, j]
            # We need to load A for all h; but L depends only on s[j], not on h specifically.
            # For each j, s[j] += A[b, h, c, j] for all h. We loop over h.
            for hh in range(H):
                a_off = b * stride_A_b + hh * stride_A_h + c * stride_A_c + j * stride_A_s
                a_val = tl.load(A_ptr + a_off)
                s[j] += a_val
        # Write L[i, j, h] = exp(s[j]) for all h; enforce lower-triangular: for j > i, set 0
        for j in range(S):
            for hh in range(H):
                # If j > i, set to 0
                if j <= i:
                    l_val = tl.exp(s[j])
                else:
                    l_val = 0.0
                l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + hh * stride_L_h
                tl.store(L_ptr + l_off, l_val)


@triton.jit
def contract_BC_to_G_kernel(
    B_ptr, C_ptr, G_ptr,
    Bsz, Csz,
    stride_B_b, stride_B_c, stride_B_s, stride_B_g, stride_B_k,
    stride_C_b, stride_C_c, stride_C_s, stride_C_g, stride_C_k,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    S: tl.constexpr,            # chunk_size
    G_CONST: tl.constexpr,      # groups (8)
    K: tl.constexpr,            # state_size (D from hidden last dim)
    num_warps=8, num_stages=2
):
    """
    Compute G[b, c, i, j, h] = sum_{g=0..G_CONST-1} sum_{k=0..K-1} C[b, c, i, g, k] * B[b, c, j, g, k].
    Grid: (Bsz, Csz, H). Loops over i, j, and accumulates over g, k.
    """
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    for i in range(S):
        for j in range(S):
            acc = 0.0
            for g in range(G_CONST):
                for k in range(K):
                    b_off = b * stride_B_b + c * stride_B_c + j * stride_B_s + g * stride_B_g + k * stride_B_k
                    c_off = b * stride_C_b + c * stride_C_c + i * stride_C_s + g * stride_C_g + k * stride_C_k
                    b_val = tl.load(B_ptr + b_off)
                    c_val = tl.load(C_ptr + c_off)
                    acc += b_val * c_val
            g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
            tl.store(G_ptr + g_off, acc)


@triton.jit
def contract_M_with_hidden_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    Bsz, Csz, S, H, D,
    stride_M_b, stride_M_c, stride_M_i, stride_M_j, stride_M_h,
    stride_hs_b, stride_hs_c, stride_hs_j, stride_hs_h, stride_hs_d,
    stride_Y_b, stride_Y_c, stride_Y_i, stride_Y_h, stride_Y_d,
    num_warps=8, num_stages=2
):
    """
    Compute Y[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden[b, c, j, h, d].
    Grid: (Bsz, Csz, S, H). Loop over d and j inside program.
    """
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    # We cannot create a vector of D here; we loop over d. Triton supports loops.
    # For robustness, we use a while loop with d initialized.
    d = 0
    while d < D:
        acc = 0.0
        for j in range(S):
            m_off = b * stride_M_b + c * stride_M_c + i * stride_M_i + j * stride_M_j + h * stride_M_h
            m_val = tl.load(M_ptr + m_off)
            hs_off = b * stride_hs_b + c * stride_hs_c + j * stride_hs_j + h * stride_hs_h + d * stride_hs_d
            hs_val = tl.load(hidden_ptr + hs_off)
            acc += m_val * hs_val
        y_off = b * stride_Y_b + c * stride_Y_c + i * stride_Y_i + h * stride_Y_h + d * stride_Y_d
        tl.store(Y_ptr + y_off, acc)
        d += 1


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-only forward:
          - compute L via Triton kernel
          - compute G via Triton kernel
          - compute M = G * L (elementwise multiply in PyTorch)
          - compute Y_diag via Triton kernel contracting M with hidden_states
          - return Y_diag in bfloat16
        """
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Inputs must be CUDA tensors"

        Bsz, Csz, S, H, D = hidden_states.shape
        assert A_cumsum.shape == (Bsz, H, Csz, S), f"Expected A_cumsum shape [B,H,C,S], got {A_cumsum.shape}"
        assert B.shape == (Bsz, Csz, S, 8, D), "Expected B shape [B,C,S,8,D]"
        assert C.shape == (Bsz, Csz, S, 8, D), "Expected C shape [B,C,S,8,D]"

        # Allocate tensors
        L = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        G = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        Y_diag = torch.empty((Bsz, Csz, S, H, D), dtype=torch.float32, device=hidden_states.device)

        # 1) Compute L via Triton kernel
        compute_L_exp_cumsum_kernel[(Bsz, Csz)](
            A_cumsum.to(torch.float32), L,
            Bsz, Csz,
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            S=S, H=H,
            num_warps=8, num_stages=2
        )

        # 2) Compute G via Triton kernel
        contract_BC_to_G_kernel[(Bsz, Csz, H)](
            B.to(torch.float32), C.to(torch.float32), G,
            Bsz, Csz,
            B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            S=S, G_CONST=8, K=D,
            num_warps=8, num_stages=2
        )

        # 3) Compute M = G * L elementwise (PyTorch)
        M = G * L  # [B, C, S, S, H], float32

        # 4) Compute Y_diag via Triton kernel contracting M with hidden_states
        # hidden_states: [B, C, S, H, D]
        # M: [B, C, S, S, H]
        # Launch grid over (Bsz, Csz, S, H)
        contract_M_with_hidden_kernel[(Bsz, Csz, S, H)](
            M, hidden_states.to(torch.float32), Y_diag,
            Bsz, Csz, S, H, D,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
            Y_diag.stride(0), Y_diag.stride(1), Y_diag.stride(2), Y_diag.stride(3), Y_diag.stride(4),
            num_warps=8, num_stages=2
        )

        # 5) Return in bfloat16
        return Y_diag.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
