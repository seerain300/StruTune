import torch
import triton
import triton.language as tl


@triton.jit
def compute_L_exp_cumsum_kernel(
    A_ptr, L_ptr,
    Bsz, Csz,
    stride_A_b, stride_A_h, stride_A_c, stride_A_s,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    S: tl.constexpr,  # e.g., 128
    H: tl.constexpr,  # e.g., 32
    num_warps=8, num_stages=2
):
    # One program per (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Loop over i, build cumsum over j for each head h, and write L[i, j, h] = exp(cumsum[j]) if j <= i
    for i in range(S):
        # cumsum vector s[j]
        s = tl.zeros([S], dtype=tl.float32)
        for j in range(S):
            for hh in range(H):
                a_off = b * stride_A_b + hh * stride_A_h + c * stride_A_c + j * stride_A_s
                a_val = tl.load(A_ptr + a_off)
                s[j] += a_val
            # store L[i, j, h] for all heads h
            for hh in range(H):
                l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + hh * stride_L_h
                if j <= i:
                    tl.store(L_ptr + l_off, tl.exp(s[j]))
                else:
                    tl.store(L_ptr + l_off, 0.0)


@triton.jit
def contract_BC_to_G_kernel(
    B_ptr, C_ptr, G_ptr,
    Bsz, Csz,
    stride_B_b, stride_B_c, stride_B_s, stride_B_g, stride_B_k,
    stride_C_b, stride_C_c, stride_C_s, stride_C_g, stride_C_k,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    S: tl.constexpr,        # 128
    G_CONST: tl.constexpr,  # 8
    K: tl.constexpr,        # 32
    num_warps=8, num_stages=2
):
    # One program per (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Accumulator G[i, j, h] per (i,j,h) across all heads (h loop below)
    for i in range(S):
        for j in range(S):
            # Accumulate over groups and state dimension
            acc = 0.0  # scalar float32 accumulator for this (i, j)
            for g in range(G_CONST):
                for k in range(K):
                    b_off = b * stride_B_b + c * stride_B_c + j * stride_B_s + g * stride_B_g + k * stride_B_k
                    c_off = b * stride_C_b + c * stride_C_c + i * stride_C_s + g * stride_C_g + k * stride_C_k
                    b_val = tl.load(B_ptr + b_off)
                    c_val = tl.load(C_ptr + c_off)
                    acc += b_val * c_val
            # Store acc to G[i, j, h] for all h (we fill all heads with same acc)
            for hh in range(H):
                g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + hh * stride_G_h
                tl.store(G_ptr + g_off, acc)


@triton.jit
def contract_M_with_hidden_reduce_j_kernel(
    G_ptr, L_ptr, hidden_ptr, Y_ptr,
    Bsz, Csz,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    stride_hs_b, stride_hs_c, stride_hs_s, stride_hs_h, stride_hs_d,
    stride_Y_b, stride_Y_c, stride_Y_i, stride_Y_h, stride_Y_d,
    S: tl.constexpr,  # 128
    H: tl.constexpr,  # 32
    D: tl.constexpr,  # 32
    num_warps=8, num_stages=2
):
    # One program per (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # For each (i, h), accumulate over j: Y[b, c, i, h, d] += sum_j (G[i, j, h] * L[i, j, h]) * hidden[b, c, j, h, d]
    for i in range(S):
        for h in range(H):
            acc = tl.zeros([D], dtype=tl.float32)  # accumulate across j for each d
            for j in range(S):
                g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
                l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
                g_val = tl.load(G_ptr + g_off)
                l_val = tl.load(L_ptr + l_off)
                prod = g_val * l_val
                # hidden[b, c, j, h, d]
                for d in range(D):
                    hs_off = b * stride_hs_b + c * stride_hs_c + j * stride_hs_s + h * stride_hs_h + d * stride_hs_d
                    hs_val = tl.load(hidden_ptr + hs_off)
                    acc[d] += prod * hs_val
            # Store acc for all d
            for d in range(D):
                y_off = b * stride_Y_b + c * stride_Y_c + i * stride_Y_i + h * stride_Y_h + d * stride_Y_d
                tl.store(Y_ptr + y_off, acc[d])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward:
          1) compute L via compute_L_exp_cumsum_kernel
          2) compute G via contract_BC_to_G_kernel
          3) compute Y_diag via contract_M_with_hidden_reduce_j_kernel
        Returns Y_diag in bfloat16 to match original behavior.
        """
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Inputs must be CUDA tensors"

        # Shapes: specialized to original constants for correctness
        Bsz, Csz, S, H, D = hidden_states.shape
        assert S == 128 and H == 32 and D == 32, "This Triton implementation expects S=128, H=32, D=32"
        assert A_cumsum.shape == (Bsz, H, Csz, S), f"Expected A_cumsum shape [B,H,C,S], got {A_cumsum.shape}"
        assert B.shape == (Bsz, Csz, S, 8, 32), "Expected B shape [B,C,S,8,32]"
        assert C.shape == (Bsz, Csz, S, 8, 32), "Expected C shape [B,C,S,8,32]"

        # Make tensors contiguous for correct stride usage
        A = A_cumsum.contiguous()
        Bt = B.contiguous()
        Ct = C.contiguous()
        Ht = hidden_states.contiguous()

        # Allocate outputs
        L = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        G = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        Y_diag = torch.empty((Bsz, Csz, S, H, D), dtype=torch.float32, device=hidden_states.device)

        # Launch kernel 1: compute L
        compute_L_exp_cumsum_kernel[(Bsz, Csz)](
            A, L,
            Bsz, Csz,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            S=S, H=H,
            num_warps=8, num_stages=2
        )

        # Launch kernel 2: compute G
        contract_BC_to_G_kernel[(Bsz, Csz)](
            Bt, Ct, G,
            Bsz, Csz,
            Bt.stride(0), Bt.stride(1), Bt.stride(2), Bt.stride(3), Bt.stride(4),
            Ct.stride(0), Ct.stride(1), Ct.stride(2), Ct.stride(3), Ct.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            S=S, G_CONST=8, K=32,
            num_warps=8, num_stages=2
        )

        # Launch kernel 3: compute Y_diag by contracting M = G * L with hidden_states over j
        contract_M_with_hidden_reduce_j_kernel[(Bsz, Csz)](
            G, L, Ht, Y_diag,
            Bsz, Csz,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            Ht.stride(0), Ht.stride(1), Ht.stride(2), Ht.stride(3), Ht.stride(4),
            Y_diag.stride(0), Y_diag.stride(1), Y_diag.stride(2), Y_diag.stride(3), Y_diag.stride(4),
            S=S, H=H, D=D,
            num_warps=8, num_stages=2
        )

        # Return in bfloat16 as original model
        return Y_diag.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
