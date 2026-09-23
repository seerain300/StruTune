import torch
import triton
import triton.language as tl


@triton.jit
def compute_L_exp_cumsum_kernel(
    A_ptr, L_ptr,
    Bsz, Csz,
    stride_A_b, stride_A_h, stride_A_c, stride_A_s,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    S: tl.constexpr,  # specialize for chunk_size = 128
    H: tl.constexpr,  # specialize for num_heads = 32
    num_warps=8, num_stages=2
):
    # Grid: (Bsz, Csz)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Build L: for each i, compute cumsum across j and write exp(cumsum) for j <= i
    for i in range(S):
        for j in range(S):
            s_j = 0.0
            for hh in range(H):
                a_off = b * stride_A_b + hh * stride_A_h + c * stride_A_c + j * stride_A_s
                a_val = tl.load(A_ptr + a_off)
                s_j += a_val
            # Write L[i, j, h] = exp(s_j) for all heads h
            for hh in range(H):
                l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + hh * stride_L_h
                l_val = tl.exp(s_j)
                tl.store(L_ptr + l_off, l_val)
        # Zero out entries where j > i (upper triangle) for all heads
        for j in range(S):
            if j > i:
                for hh in range(H):
                    l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + hh * stride_L_h
                    tl.store(L_ptr + l_off, 0.0)


@triton.jit
def contract_BC_to_G_kernel(
    B_ptr, C_ptr, G_ptr,
    Bsz, Csz,
    stride_B_b, stride_B_c, stride_B_s, stride_B_g, stride_B_k,
    stride_C_b, stride_C_c, stride_C_s, stride_C_g, stride_C_k,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    S: tl.constexpr,   # 128
    G_CONST: tl.constexpr,  # 8
    K: tl.constexpr,  # 32
    num_warps=8, num_stages=2
):
    # Grid: (Bsz, Csz)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Loop over i and j
    for i in range(S):
        for j in range(S):
            acc = 0.0
            # Accumulate over groups and state
            for g in range(G_CONST):
                for k in range(K):
                    b_off = b * stride_B_b + c * stride_B_c + j * stride_B_s + g * stride_B_g + k * stride_B_k
                    c_off = b * stride_C_b + c * stride_C_c + i * stride_C_s + g * stride_C_g + k * stride_C_k
                    b_val = tl.load(B_ptr + b_off)
                    c_val = tl.load(C_ptr + c_off)
                    acc += b_val * c_val
            # Store G[i, j, h] for all heads h
            for hh in range(H):
                g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + hh * stride_G_h
                tl.store(G_ptr + g_off, acc)


@triton.jit
def contract_M_with_hidden_per_d_kernel(
    G_ptr, L_ptr, hidden_ptr, Y_ptr,
    Bsz, Csz,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    stride_hs_b, stride_hs_c, stride_hs_s, stride_hs_h, stride_hs_d,
    stride_Y_b, stride_Y_c, stride_Y_i, stride_Y_h, stride_Y_d,
    S: tl.constexpr,   # 128
    H: tl.constexpr,   # 32
    d_idx: tl.constexpr,  # 0..31
    num_warps=8, num_stages=2
):
    # Grid: (Bsz, Csz, S, H)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    acc = 0.0
    # Loop over j and compute sum_j G[i, j, h] * L[i, j, h] * hidden[b, c, j, h, d]
    for j in range(S):
        g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
        l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
        g_val = tl.load(G_ptr + g_off)
        l_val = tl.load(L_ptr + l_off)
        hs_off = b * stride_hs_b + c * stride_hs_c + j * stride_hs_s + h * stride_hs_h + d_idx * stride_hs_d
        hs_val = tl.load(hidden_ptr + hs_off)
        acc += g_val * l_val * hs_val

    y_off = b * stride_Y_b + c * stride_Y_c + i * stride_Y_i + h * stride_Y_h + d_idx * stride_Y_d
    tl.store(Y_ptr + y_off, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward that:
          1) computes L via compute_L_exp_cumsum_kernel using actual C_in
          2) computes G via contract_BC_to_G_kernel using actual C_in
          3) computes Y_diag via contract_M_with_hidden_per_d_kernel using actual C_in
        Returns Y_diag with shape [B, C_in, S, H, D], matching the original model.
        """
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Inputs must be CUDA tensors"

        # Shapes: specialize for the original model axes
        Bsz, Csz, S, H, D = hidden_states.shape
        assert S == 128 and H == 32 and D == 32, "This Triton implementation specializes for S=128, H=32, D=32"
        assert A_cumsum.shape == (Bsz, H, Csz, S), f"Expected A_cumsum shape [B,H,C,S], got {A_cumsum.shape}"
        assert B.shape == (Bsz, Csz, S, 8, 32), "Expected B shape [B,C,S,8,32]"
        assert C.shape == (Bsz, Csz, S, 8, 32), "Expected C shape [B,C,S,8,32]"

        # Ensure contiguous tensors
        A = A_cumsum.contiguous()
        Bn = B.contiguous()
        Cn = C.contiguous()
        hidden = hidden_states.contiguous()

        # Allocate outputs
        L = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden.device)
        G = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden.device)
        Y_diag = torch.empty((Bsz, Csz, S, H, D), dtype=torch.float32, device=hidden.device)

        # Launch kernel 1: compute L using actual Csz (num_chunks from input)
        compute_L_exp_cumsum_kernel[(Bsz, Csz)](
            A, L,
            Bsz, Csz,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            S=128, H=32,
            num_warps=8, num_stages=2
        )

        # Launch kernel 2: compute G using actual Csz
        contract_BC_to_G_kernel[(Bsz, Csz)](
            Bn, Cn, G,
            Bsz, Csz,
            Bn.stride(0), Bn.stride(1), Bn.stride(2), Bn.stride(3), Bn.stride(4),
            Cn.stride(0), Cn.stride(1), Cn.stride(2), Cn.stride(3), Cn.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            S=128, G_CONST=8, K=32,
            num_warps=8, num_stages=2
        )

        # Launch kernel 3: per-d contraction over D using actual Csz
        for d_idx in range(D):
            contract_M_with_hidden_per_d_kernel[(Bsz, Csz, S, H)](
                G, L, hidden, Y_diag,
                Bsz, Csz,
                G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
                L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
                hidden.stride(0), hidden.stride(1), hidden.stride(2), hidden.stride(3), hidden.stride(4),
                Y_diag.stride(0), Y_diag.stride(1), Y_diag.stride(2), Y_diag.stride(3), Y_diag.stride(4),
                S=128, H=32, d_idx=d_idx,
                num_warps=8, num_stages=2
            )

        # Return in bfloat16 to match original behavior (compute is in fp32)
        return Y_diag.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
