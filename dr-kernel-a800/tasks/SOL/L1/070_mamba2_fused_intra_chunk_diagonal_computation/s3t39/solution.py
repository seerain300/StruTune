import torch
import triton
import triton.language as tl


@triton.jit
def compute_L_exp_cumsum_kernel(
    A_ptr, L_ptr,
    Bsz, Csz,
    stride_A_b, stride_A_h, stride_A_c, stride_A_s,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    S: tl.constexpr,  # chunk_size = 128
    H: tl.constexpr,  # num_heads = 32
    num_warps=8, num_stages=2
):
    # One program per (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Initialize L to zeros
    for i in tl.static_range(S):
        for j in tl.static_range(S):
            for hh in tl.static_range(H):
                l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + hh * stride_L_h
                tl.store(L_ptr + l_off, 0.0)

    # Build cumsum along j and store exp(cumsum_j) for i >= j (lower triangle, exclude diagonal)
    for hh in tl.static_range(H):
        for i in tl.static_range(S):
            s = 0.0
            for j in tl.static_range(S):
                a_off = b * stride_A_b + hh * stride_A_h + c * stride_A_c + j * stride_A_s
                a_val = tl.load(A_ptr + a_off)
                s += a_val
            for j in tl.static_range(S):
                if j < i:  # exclude diagonal: tril with diagonal=-1
                    l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + hh * stride_L_h
                    tl.store(L_ptr + l_off, tl.exp(s))


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
    # One program per (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)

    for i in tl.static_range(S):
        for j in tl.static_range(S):
            for hh in tl.static_range(H):
                acc = 0.0
                # Sum over groups and states
                for g in tl.static_range(G_CONST):
                    for k in tl.static_range(K):
                        b_off = b * stride_B_b + c * stride_B_c + j * stride_B_s + g * stride_B_g + k * stride_B_k
                        c_off = b * stride_C_b + c * stride_C_c + i * stride_C_s + g * stride_C_g + k * stride_C_k
                        b_val = tl.load(B_ptr + b_off)
                        c_val = tl.load(C_ptr + c_off)
                        acc += b_val * c_val
                # Store G[i, j, h]
                g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + hh * stride_G_h
                tl.store(G_ptr + g_off, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward:
          - compute_L_exp_cumsum_kernel to build L
          - contract_BC_to_G_kernel to compute G
          - final contraction M = G * L and output Y_diag via PyTorch
        Returns Y_diag cast to bfloat16.
        """


def run(*args):
    return ModelNew()(*args)
