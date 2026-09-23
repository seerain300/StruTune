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
    # One program per (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # For each i, build cumsum across j and write L[i, j, h] with lower-triangular mask (i >= j)
    for i in range(S):
        # Compute cumsum vector s[j] = sum_{t <= j} A[b, h, c, t]
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
        # Explicitly zero out for j > i to enforce triangular mask
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
    S: tl.constexpr,  # 128
    G_CONST: tl.constexpr,  # 8
    K: tl.constexpr,  # 32
    num_warps=8, num_stages=2
):
    # Grid: (Bsz, Csz)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Loop over i and j, accumulate G[i, j, h]
    for i in range(S):
        for j in range(S):
            acc = 0.0
            for g in range(G_CONST):
                for k in range(K):
                    # B[b, c, j, g, k]
                    b_off = b * stride_B_b + c * stride_B_c + j * stride_B_s + g * stride_B_g + k * stride_B_k
                    b_val = tl.load(B_ptr + b_off)
                    # C[b, c, i, g, k]
                    c_off = b * stride_C_b + c * stride_C_c + i * stride_C_s + g * stride_C_g + k * stride_C_k
                    c_val = tl.load(C_ptr + c_off)
                    acc += b_val * c_val
            # Store G[i, j, h] for all heads h (H=32)
            for hh in range(32):
                g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + hh * stride_G_h
                tl.store(G_ptr + g_off, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward that:
          1) computes L via compute_L_exp_cumsum_kernel
          2) computes G via contract_BC_to_G_kernel
          3) computes Y_diag via PyTorch contraction of M = G * L with hidden_states over the j dimension
        Returns: [B, C, S, H, D] in bfloat16 to match original behavior.
        """
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Inputs must be CUDA tensors"

        # Shapes and specialization (these constants match the original model)
        Bsz, Csz, S, H, D = hidden_states.shape
        assert S == 128 and H == 32 and D == 32, "This Triton implementation specializes for S=128, H=32, D=32"
        assert A_cumsum.shape == (Bsz, H, Csz, S), f"Expected A_cumsum shape [B,H,C,S], got {A_cumsum.shape}"
        assert B.shape == (Bsz, Csz, S, 8, 32), "Expected B shape [B,C,S,8,32]"
        assert C.shape == (Bsz, Csz, S, 8, 32), "Expected C shape [B,C,S,8,32]"

        # Ensure contiguous
        A = A_cumsum.contiguous()
        Bc = B.contiguous()
        Cc = C.contiguous()
        hidden = hidden_states.contiguous()

        # Allocate outputs (float32 for compute)
        L = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden.device)
        G = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden.device)

        # Launch kernel 1: compute L
        compute_L_exp_cumsum_kernel[(Bsz, Csz)](
            A, L,
            Bsz, Csz,
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            S=128, H=32,
            num_warps=8, num_stages=2
        )

        # Launch kernel 2: compute G from B and C
        contract_BC_to_G_kernel[(Bsz, Csz)](
            Bc, Cc, G,
            Bsz, Csz,
            Bc.stride(0), Bc.stride(1), Bc.stride(2), Bc.stride(3), Bc.stride(4),
            Cc.stride(0), Cc.stride(1), Cc.stride(2), Cc.stride(3), Cc.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            S=128, G_CONST=8, K=32,
            num_warps=8, num_stages=2
        )

        # Compute M = G * L elementwise (PyTorch)
        M = G * L  # [B, C, S, S, H], float32

        # Compute Y_diag: sum_j M[:, :, i, j, h] * hidden[:, :, j, h, :] -> [B, C, S, H, D]
        Y_diag = torch.empty((Bsz, Csz, S, H, D), dtype=torch.float32, device=hidden.device)
        for b in range(Bsz):
            for c in range(Csz):
                for i in range(S):
                    for h in range(H):
                        # Sum over j: Y_diag[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden[b, c, j, h, d]
                        # hidden slice: [S] along j, [H] along h, [D] along d
                        for d in range(D):
                            sum_val = 0.0
                            for j in range(S):
                                m_val = M[b, c, i, j, h].item()  # scalar
                                h_val = hidden[b, c, j, h, d].item()
                                sum_val += m_val * h_val
                            Y_diag[b, c, i, h, d] = sum_val

        # Convert to bfloat16 to match original function
        Y_diag = Y_diag.to(torch.bfloat16)
        return Y_diag


def run(*args):
    return ModelNew()(*args)
