import torch
import triton
import triton.language as tl


@triton.jit
def compute_L_exp_cumsum_kernel(
    A_ptr, L_ptr,
    Bsz, Csz,
    stride_A_b, stride_A_h, stride_A_c, stride_A_s,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    S: tl.constexpr,  # chunk_size, e.g., 128
    H: tl.constexpr,  # num_heads, e.g., 32
    num_warps=8, num_stages=2
):
    """
    Compute L[b, c, i, j, h] = exp(sum_{t<=j} A[b, h, c, t]) for j <= i, else 0.
    Grid: (Bsz, Csz). Loops over i and j vectors and h.
    """
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Loop over i
    for i in range(S):
        # Maintain cumsum vector s[j] for all j
        s = tl.zeros((S,), dtype=tl.float32)
        # Loop over j to build cumsum
        for j in range(S):
            # For each head h, s[j] += A[b, h, c, j]
            for hh in range(H):
                a_off = b * stride_A_b + hh * stride_A_h + c * stride_A_c + j * stride_A_s
                a_val = tl.load(A_ptr + a_off)
                s[j] += a_val
        # Store L[i, j, h] = exp(s[j]) if j <= i else 0
        for j in range(S):
            write_mask = j <= i  # lower-triangular inclusive diagonal
            for hh in range(H):
                l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + hh * stride_L_h
                l_val = tl.exp(s[j])
                tl.store(L_ptr + l_off, l_val, mask=write_mask)


@triton.jit
def contract_BC_to_G_kernel(
    B_ptr, C_ptr, G_ptr,
    Bsz, Csz,
    stride_B_b, stride_B_c, stride_B_s, stride_B_g, stride_B_k,
    stride_C_b, stride_C_c, stride_C_s, stride_C_g, stride_C_k,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    S: tl.constexpr,            # 128
    G_CONST: tl.constexpr,      # 8
    K: tl.constexpr,            # 32
    num_warps=8, num_stages=2
):
    """
    Compute G[b, c, i, j, h] = sum_{g=0..G_CONST-1} sum_{k=0..K-1} C[b, c, i, g, k] * B[b, c, j, g, k].
    Grid: (Bsz, Csz). Loops over i, j, h; accumulates over g, k.
    """
    b = tl.program_id(0)
    c = tl.program_id(1)

    for i in range(S):
        for j in range(S):
            # Accumulator for a given head h (we'll loop h)
            for h in range(S):  # assuming H == S
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
    Bsz, Csz,
    stride_M_b, stride_M_c, stride_M_i, stride_M_j, stride_M_h,
    stride_hs_b, stride_hs_c, stride_hs_s, stride_hs_h, stride_hs_d,
    stride_Y_b, stride_Y_c, stride_Y_i, stride_Y_h, stride_Y_d,
    S: tl.constexpr,  # 128
    H: tl.constexpr,  # 32
    D: tl.constexpr,  # 32
    num_warps=8, num_stages=2
):
    """
    Compute Y_diag[b, c, i, h, d] = sum_{j=0..S-1} M[b, c, i, j, h] * hidden[b, c, j, h, d]
    Launch grid over (Bsz * Csz * S * H). Each program handles one (b, c, i, h, d) and loops over j.
    """
    # Grid: programs over (Bsz*Csz*S*H)
    pid = tl.program_id(0)
    total = Bsz * Csz * S * H
    # Recover indices
    tmp = pid
    i = tmp % S
    tmp = tmp // S
    h = tmp % H
    tmp = tmp // H
    c = tmp % Csz
    b = tmp // Csz

    # Accumulator for this (b, c, i, h, d)
    acc = 0.0
    for j in range(S):
        m_off = b * stride_M_b + c * stride_M_c + i * stride_M_i + j * stride_M_j + h * stride_M_h
        h_off = b * stride_hs_b + c * stride_hs_c + j * stride_hs_s + h * stride_hs_h
        # d loop
        for d in range(D):
            m_val = tl.load(M_ptr + m_off)
            hs_val = tl.load(hidden_ptr + h_off + d * stride_hs_d)
            acc += m_val * hs_val
    # Store result
    y_off = b * stride_Y_b + c * stride_Y_c + i * stride_Y_i + h * stride_Y_h
    tl.store(Y_ptr + y_off, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward that:
          1) computes L via compute_L_exp_cumsum_kernel
          2) computes G via contract_BC_to_G_kernel
          3) computes Y_diag via contract_M_with_hidden_kernel
        Returns: Y_diag with dtype bfloat16 and shape [B, C, S, H, D].
        """
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "All inputs must be CUDA tensors"

        # Ensure contiguity (though not strictly necessary if we use strides)
        hidden_states = hidden_states.contiguous()
        A_cumsum = A_cumsum.contiguous()
        B = B.contiguous()
        C = C.contiguous()

        Bsz, Csz, S, H, D = hidden_states.shape
        # Original model uses H == 32, S == 128, D == 32, G_CONST == 8, K == 32.
        # We will specialize kernels for these constants (common case). If different, the code still runs via Python loops,
        # but Triton performance is best with these sizes. We assert for clarity.
        assert H == 32 and S == 128 and D == 32, "This Triton implementation specializes for H=32, S=128, D=32"
        G_CONST = 8
        K = 32

        # Allocate outputs
        L = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        G = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        Y_diag = torch.empty((Bsz, Csz, S, H, D), dtype=torch.float32, device=hidden_states.device)

        # Launch kernel 1: compute L
        compute_L_exp_cumsum_kernel[(Bsz, Csz)](
            A_cumsum, L,
            Bsz, Csz,
            stride_A_b=A_cumsum.stride(0), stride_A_h=A_cumsum.stride(1), stride_A_c=A_cumsum.stride(2), stride_A_s=A_cumsum.stride(3),
            stride_L_b=L.stride(0), stride_L_c=L.stride(1), stride_L_i=L.stride(2), stride_L_j=L.stride(3), stride_L_h=L.stride(4),
            S=S, H=H,
            num_warps=8, num_stages=2
        )

        # Launch kernel 2: compute G from B and C
        contract_BC_to_G_kernel[(Bsz, Csz)](
            B, C, G,
            Bsz, Csz,
            stride_B_b=B.stride(0), stride_B_c=B.stride(1), stride_B_s=B.stride(2), stride_B_g=B.stride(3), stride_B_k=B.stride(4),
            stride_C_b=C.stride(0), stride_C_c=C.stride(1), stride_C_s=C.stride(2), stride_C_g=C.stride(3), stride_C_k=C.stride(4),
            stride_G_b=G.stride(0), stride_G_c=G.stride(1), stride_G_i=G.stride(2), stride_G_j=G.stride(3), stride_G_h=G.stride(4),
            S=S, G_CONST=G_CONST, K=K,
            num_warps=8, num_stages=2
        )

        # Elementwise multiply to form M
        # Note: Triton doesn't have a generic broadcasted multiply reduction kernel here; we do it in PyTorch
        # M = G * L
        M = G * L  # elementwise

        # Launch kernel 3: contract M with hidden_states along j to form Y_diag
        # Grid size: Bsz * Csz * S * H
        total_programs = Bsz * Csz * S * H
        contract_M_with_hidden_kernel[(total_programs,)](
            M, hidden_states, Y_diag,
            Bsz, Csz,
            stride_M_b=M.stride(0), stride_M_c=M.stride(1), stride_M_i=M.stride(2), stride_M_j=M.stride(3), stride_M_h=M.stride(4),
            stride_hs_b=hidden_states.stride(0), stride_hs_c=hidden_states.stride(1), stride_hs_s=hidden_states.stride(2),
            stride_hs_h=hidden_states.stride(3), stride_hs_d=hidden_states.stride(4),
            stride_Y_b=Y_diag.stride(0), stride_Y_c=Y_diag.stride(1), stride_Y_i=Y_diag.stride(2), stride_Y_h=Y_diag.stride(3),
            stride_Y_d=Y_diag.stride(4),
            S=S, H=H, D=D,
            num_warps=8, num_stages=2
        )

        # Return in bfloat16 as original code
        return Y_diag.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
