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
    num_warps=4, num_stages=2
):
    # Grid: (Bsz, Csz, S, S)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)

    # For each head h, compute cumsum across j up to j, then write L[i, j, h] = exp(cumsum[j]) if j <= i, else 0.
    for hh in range(H):
        s_j = 0.0
        # Loop over j to compute cumsum of A[b, hh, c, j]
        for jj in range(S):
            a_off = b * stride_A_b + hh * stride_A_h + c * stride_A_c + jj * stride_A_s
            a_val = tl.load(A_ptr + a_off)
            s_j += a_val
        l_val = tl.exp(s_j)
        # Store to L[i, j, h]
        if j <= i:
            l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + hh * stride_L_h
            tl.store(L_ptr + l_off, l_val)
        else:
            l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + hh * stride_L_h
            tl.store(L_ptr + l_off, 0.0)


@triton.jit
def contract_BC_to_G_kernel(
    B_ptr, C_ptr, G_ptr,
    Bsz, Csz,
    stride_B_b, stride_B_c, stride_B_s, stride_B_g, stride_B_k,
    stride_C_b, stride_C_c, stride_C_s, stride_C_g, stride_C_k,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    S: tl.constexpr,  # chunk_size
    G_CONST: tl.constexpr,  # groups (8)
    K: tl.constexpr,  # state_size (32)
    num_warps=4, num_stages=2
):
    # Grid: (Bsz, Csz, S, S)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)

    # Compute G[i, j, h] for all h
    for hh in range(H):
        acc = 0.0
        for g in range(G_CONST):
            for k in range(K):
                b_off = b * stride_B_b + c * stride_B_c + j * stride_B_s + g * stride_B_g + k * stride_B_k
                c_off = b * stride_C_b + c * stride_C_c + i * stride_C_s + g * stride_C_g + k * stride_C_k
                b_val = tl.load(B_ptr + b_off)
                c_val = tl.load(C_ptr + c_off)
                acc += b_val * c_val
        g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + hh * stride_G_h
        tl.store(G_ptr + g_off, acc)


@triton.jit
def contract_M_with_hidden_kernel(
    G_ptr, L_ptr, hidden_ptr, Y_ptr,
    Bsz, Csz, S,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    stride_hs_b, stride_hs_c, stride_hs_s, stride_hs_h, stride_hs_d,
    stride_Y_b, stride_Y_c, stride_Y_i, stride_Y_h, stride_Y_d,
    H: tl.constexpr,  # num_heads (32)
    D: tl.constexpr,  # head_dim (32)
    num_warps=4, num_stages=2
):
    # Grid: (Bsz, Csz, S, H, D)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    d = tl.program_id(4)

    acc = 0.0
    for j in range(S):
        g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
        l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
        g_val = tl.load(G_ptr + g_off)
        l_val = tl.load(L_ptr + l_off)
        hs_off = b * stride_hs_b + c * stride_hs_c + j * stride_hs_s + h * stride_hs_h + d * stride_hs_d
        hs_val = tl.load(hidden_ptr + hs_off)
        acc += g_val * l_val * hs_val

    y_off = b * stride_Y_b + c * stride_Y_c + i * stride_Y_i + h * stride_Y_h + d * stride_Y_d
    tl.store(Y_ptr + y_off, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward:
          1) compute L via compute_L_exp_cumsum_kernel
          2) compute G via contract_BC_to_G_kernel
          3) compute Y_diag via contract_M_with_hidden_kernel
        Returns Y_diag in bfloat16 to match original behavior.
        """
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Inputs must be CUDA tensors"
        hidden_states = hidden_states.contiguous()
        A_cumsum = A_cumsum.contiguous()
        B = B.contiguous()
        C = C.contiguous()

        # Shapes: specialize for S=128, H=32, D=32, G_CONST=8, K=32 as in original
        Bsz, Csz, S, H, D = hidden_states.shape
        assert S == 128 and H == 32 and D == 32, "This Triton implementation specializes for S=128, H=32, D=32"
        assert A_cumsum.shape == (Bsz, H, Csz, S), f"Expected A_cumsum shape [B,H,C,S], got {A_cumsum.shape}"
        assert B.shape == (Bsz, Csz, S, 8, 32), "Expected B shape [B,C,S,8,32]"
        assert C.shape == (Bsz, Csz, S, 8, 32), "Expected C shape [B,C,S,8,32]"

        # Allocate outputs
        L = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        G = torch.empty


def run(*args):
    return ModelNew()(*args)
