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
        for j in range(S):
            s_j = 0.0
            # sum across heads h
            for hh in range(H):
                a_off = b * stride_A_b + hh * stride_A_h + c * stride_A_c + j * stride_A_s
                a_val = tl.load(A_ptr + a_off)
                s_j += a_val
            # write L[i, j, h] = exp(s_j) for all heads h (i >= j enforces triangular via masking)
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
    S: tl.constexpr,      # 128
    G_CONST: tl.constexpr, # 8
    K: tl.constexpr,      # 32
    H: tl.constexpr,      # 32
    num_warps=8, num_stages=2
):
    # One program per (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Accumulator per (i, j, h)
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
            # Write G[i, j, h] for all heads h
            for hh in range(H):
                g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + hh * stride_G_h
                tl.store(G_ptr + g_off, acc)


@triton.jit
def contract_M_with_hidden_kernel(
    G_ptr, L_ptr, hidden_ptr, Y_ptr,
    Bsz, Csz,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    stride_H_b, stride_H_c, stride_H_s, stride_H_h, stride_H_d,
    stride_Y_b, stride_Y_c, stride_Y_i, stride_Y_h, stride_Y_d,
    S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    num_warps=8, num_stages=2
):
    # Grid: (Bsz, Csz, S, H)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    # Accumulator for output [i, h, d]
    # We'll write across D and store into Y
    for d in range(D):
        # Reduce over j
        acc = 0.0
        for j in range(S):
            g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
            l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
            h_off = b * stride_H_b + c * stride_H_c + j * stride_H_s + h * stride_H_h + d * stride_H_d
            g_val = tl.load(G_ptr + g_off)
            l_val = tl.load(L_ptr + l_off)
            h_val = tl.load(hidden_ptr + h_off)
            acc += g_val * l_val * h_val
        y_off = b * stride_Y_b + c * stride_Y_c + i * stride_Y_i + h * stride_Y_h + d * stride_Y_d
        tl.store(Y_ptr + y_off, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward that:
          1) computes L via compute_L_exp_cumsum_kernel
          2) computes G via contract_BC_to_G_kernel
          3) computes Y_diag by contracting M = G * L with hidden in Triton kernel.
        """
        # Constants assumed by original code
        S = 128  # chunk_size
        H = 32   # num_heads
        G_CONST = 8
        K = 32

        Bsz, Csz, _, _, _ = hidden_states.shape
        # Shapes must match original assumptions
        assert A_cumsum.shape == (Bsz, H, Csz, S), f"Expected A_cumsum shape [B,H,C,S], got {A_cumsum.shape}"
        assert B.shape == (Bsz, Csz, S, G_CONST, K), "Expected B shape [B,C,S,8,32]"
        assert C.shape == (Bsz, Csz, S, G_CONST, K), "Expected C shape [B,C,S,8,32]"

        # Ensure contiguous
        A = A_cumsum.contiguous()
        B_c = B.contiguous()
        C_c = C.contiguous()
        hidden = hidden_states.contiguous()

        # Allocate outputs
        L = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden.device)
        G = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden.device)
        D = hidden.shape[-1]
        Y = torch.empty((Bsz, Csz, S, H, D), dtype=torch.float32, device=hidden.device)

        # Launch kernel 1: compute L
        compute_L_exp_cumsum_kernel[(Bsz, Csz)](
            A, L,
            Bsz, Csz,
            stride_A_b=A.stride(0), stride_A_h=A.stride(1), stride_A_c=A.stride(2), stride_A_s=A.stride(3),
            stride_L_b=L.stride(0), stride_L_c=L.stride(1), stride_L_i=L.stride(2), stride_L_j=L.stride(3), stride_L_h=L.stride(4),
            S=S, H=H, num_warps=8, num_stages=2
        )

        # Launch kernel 2: compute G
        contract_BC_to_G_kernel[(Bsz, Csz)](
            B_c, C_c, G,
            Bsz, Csz,
            stride_B_b=B_c.stride(0), stride_B_c=B_c.stride(1), stride_B_s=B_c.stride(2), stride_B_g=B_c.stride(3), stride_B_k=B_c.stride(4),
            stride_C_b=C_c.stride(0), stride_C_c=C_c.stride(1), stride_C_s=C_c.stride(2), stride_C_g=C_c.stride(3), stride_C_k=C_c.stride(4),
            stride_G_b=G.stride(0), stride_G_c=G.stride(1), stride_G_i=G.stride(2), stride_G_j=G.stride(3), stride_G_h=G.stride(4),
            S=S, G_CONST=G_CONST, K=K, H=H, num_warps=8, num_stages=2
        )

        # Launch kernel 3: contract M = G * L with hidden over j to produce Y_diag
        contract_M_with_hidden_kernel[(Bsz, Csz, S, H)](
            G, L, hidden, Y,
            Bsz, Csz,
            stride_G_b=G.stride(0), stride_G_c=G.stride(1), stride_G_i=G.stride(2), stride_G_j=G.stride(3), stride_G_h=G.stride(4),
            stride_L_b=L.stride(0), stride_L_c=L.stride(1), stride_L_i=L.stride(2), stride_L_j=L.stride(3), stride_L_h=L.stride(4),
            stride_H_b=hidden.stride(0), stride_H_c=hidden.stride(1), stride_H_s=hidden.stride(2), stride_H_h=hidden.stride(3), stride_H_d=hidden.stride(4),
            stride_Y_b=Y.stride(0), stride_Y_c=Y.stride(1), stride_Y_i=Y.stride(2), stride_Y_h=Y.stride(3), stride_Y_d=Y.stride(4),
            S=S, H=H, D=D, num_warps=8, num_stages=2
        )

        # Return in bfloat16 to match original output dtype
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
