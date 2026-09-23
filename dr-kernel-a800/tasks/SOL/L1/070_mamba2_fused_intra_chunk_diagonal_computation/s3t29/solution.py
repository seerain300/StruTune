import torch
import triton
import triton.language as tl


@triton.jit
def contract_BC_to_G_kernel(
    B_ptr, C_ptr, G_ptr,
    Bsz, Csz,
    stride_B_b, stride_B_c, stride_B_s, stride_B_g, stride_B_k,
    stride_C_b, stride_C_c, stride_C_s, stride_C_g, stride_C_k,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    S: tl.constexpr,       # 128
    G_CONST: tl.constexpr, # 8
    K: tl.constexpr,       # 32
    H: tl.constexpr,       # 32
    num_warps=8, num_stages=2
):
    # One program per (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Accumulator for G[i, j, h] across all h (we write per h)
    # We'll iterate j, then i; for each (i, j), accumulate over g and k.
    for i in range(S):
        for j in range(S):
            acc = 0.0
            # Sum over groups g and state k
            for g in range(G_CONST):
                for k in range(K):
                    b_off = b * stride_B_b + c * stride_B_c + j * stride_B_s + g * stride_B_g + k * stride_B_k
                    c_off = b * stride_C_b + c * stride_C_c + i * stride_C_s + g * stride_C_g + k * stride_C_k
                    b_val = tl.load(B_ptr + b_off)
                    c_val = tl.load(C_ptr + c_off)
                    acc += b_val * c_val
            # Write G[i, j, h] = acc for all h
            for h in range(H):
                g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
                tl.store(G_ptr + g_off, acc)


@triton.jit
def contract_M_with_hidden_kernel(
    G_ptr, L_ptr, hidden_ptr, Y_ptr,
    Bsz, Csz, H, D,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    stride_hs_b, stride_hs_c, stride_hs_s, stride_hs_h, stride_hs_d,
    stride_Y_b, stride_Y_c, stride_Y_i, stride_Y_h, stride_Y_d,
    S: tl.constexpr,  # 128
    H_const: tl.constexpr,  # 32
    D_const: tl.constexpr,  # 32
    num_warps=8, num_stages=2
):
    # Grid: (Bsz, Csz, S, H_const, D_const)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    d = tl.program_id(4)

    # Compute Y[b, c, i, h, d] = sum_j (G[b, c, i, j, h] * L[b, c, i, j, h]) * hidden[b, c, j, h, d]
    total = 0.0
    for j in range(S):
        g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
        l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
        g_val = tl.load(G_ptr + g_off)
        l_val = tl.load(L_ptr + l_off)
        hs_off = b * stride_hs_b + c * stride_hs_c + j * stride_hs_s + h * stride_hs_h + d * stride_hs_d
        hs_val = tl.load(hidden_ptr + hs_off)
        total += g_val * l_val * hs_val

    y_off = b * stride_Y_b + c * stride_Y_c + i * stride_Y_i + h * stride_Y_h + d * stride_Y_d
    tl.store(Y_ptr + y_off, total)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward:
          - Computes G via contract_BC_to_G_kernel
          - Computes Y_diag via contract_M_with_hidden_kernel using L = torch.exp(torch.cumsum(A_cumsum, dim=-1))
        Returns Y in bfloat16.
        """
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Inputs must be CUDA tensors"

        # Ensure contiguity
        A_cumsum = A_cumsum.contiguous()
        B = B.contiguous()
        C = C.contiguous()
        hidden_states = hidden_states.contiguous()

        Bsz, Csz, S, H, D = hidden_states.shape
        assert S == 128 and H == 32 and D == 32, "This Triton implementation specializes for S=128, H=32, D=32"
        assert A_cumsum.shape == (Bsz, H, Csz, S), f"Expected A_cumsum shape [B,H,C,S], got {A_cumsum.shape}"
        assert B.shape == (Bsz, Csz, S, 8, 32), "Expected B shape [B,C,S,8,32]"
        assert C.shape == (Bsz, Csz, S, 8, 32), "Expected C shape [B,C,S,8,32]"

        # Allocate intermediate G and output Y
        G = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        Y = torch.empty((Bsz, Csz, S, H, D), dtype=torch.float32, device=hidden_states.device)

        # Launch kernel 1: contract B and C to G
        contract_BC_to_G_kernel[(Bsz, Csz)](
            B, C, G,
            Bsz, Csz,
            B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            S=128, G_CONST=8, K=32, H=32,
            num_warps=8, num_stages=2
        )

        # Compute L using PyTorch (to ensure correctness), then launch kernel 2
        # L: [B, C, S, S, H] = exp(cumsum along S for each (b, h, c))
        # Expand over heads (all heads share the same L per (b, c, i, j))
        A_cumsum_expanded = A_cumsum.unsqueeze(-1).unsqueeze(-1)  # [B, H, C, S, 1, 1]
        # Cumsum along S dimension
        A_cumsum_s = torch.cumsum(A_cumsum_expanded, dim=-1)      # [B, H, C, S, 1, 1]
        L = torch.exp(A_cumsum_s)                                 # [B, H, C, S, 1, 1]
        # Remove singleton dims to match original L shape [B, C, S, S, H]
        L = L.squeeze(-1).squeeze(-1).permute(0, 2, 3, 1).permute(0, 2, 4, 3, 1)  # [B, C, S, S, H]

        # Launch kernel 2: contract M = G * L with hidden to get Y_diag
        contract_M_with_hidden_kernel[(Bsz, Csz, S, H, D)](
            G, L, hidden_states, Y,
            Bsz, Csz, H, D,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            S=128, H_const=32, D_const=32,
            num_warps=8, num_stages=2
        )

        # Return in bfloat16
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
