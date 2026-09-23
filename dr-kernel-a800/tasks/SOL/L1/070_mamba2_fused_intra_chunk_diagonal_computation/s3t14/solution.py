import torch
import triton
import triton.language as tl


@triton.jit
def compute_L_exp_cumsum_kernel(
    A_ptr, L_ptr,
    Bsz, Csz,
    stride_A_b, stride_A_h, stride_A_c, stride_A_s,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    S: tl.constexpr,  # 128
    H: tl.constexpr,  # 32
    num_warps=8, num_stages=2
):
    # Grid: (Bsz, Csz)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # For each head h, compute cumsum across j and fill L[i, j, h] with lower-triangular mask
    for h in range(H):
        # s[j] = sum_{t<=j} A[b, h, c, t]
        s = tl.zeros([S], dtype=tl.float32)
        for j in range(S):
            # A[b, h, c, j]
            a_off = b * stride_A_b + h * stride_A_h + c * stride_A_c + j * stride_A_s
            a_val = tl.load(A_ptr + a_off)
            s[j] = a_val
            # Now propagate cumsum: s[j] accumulates A[b, h, c, j]
            # For future j, s[j] += A[b, h, c, j]
        # Fill L[i, j, h] for i in [0..S-1]
        for i in range(S):
            for j in range(S):
                if j <= i:
                    l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
                    l_val = tl.exp(s[j])  # float32
                    tl.store(L_ptr + l_off, l_val)
                else:
                    l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
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

    # Loop over heads h (H=32)
    for h in range(32):
        # Accumulator for G[i, j, h]
        for i in range(S):
            for j in range(S):
                acc = 0.0
                # Sum over groups and states: B[b, c, j, g, k] * C[b, c, i, g, k]
                # Note: H=32 and groups repeat by 4, so h maps to g in {0,1,2,3}
                # Each group contributes to 4 heads: h, h+1, h+2, h+3 (wrap if h+3 >= 32)
                # Here, we explicitly assign groups 0..7 and sum their contributions.
                for g in range(G_CONST):
                    for k in range(K):
                        # B index
                        b_off = b * stride_B_b + c * stride_B_c + j * stride_B_s + g * stride_B_g + k * stride_B_k
                        # C index
                        c_off = b * stride_C_b + c * stride_C_c + i * stride_C_s + g * stride_C_g + k * stride_C_k
                        b_val = tl.load(B_ptr + b_off)
                        c_val = tl.load(C_ptr + c_off)
                        acc += b_val * c_val
                # Store G[i, j, h]
                g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
                tl.store(G_ptr + g_off, acc)


@triton.jit
def compute_Y_diag_kernel(
    G_ptr, L_ptr, hidden_ptr, Y_ptr,
    Bsz, Csz, S, H, D,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    stride_hs_b, stride_hs_c, stride_hs_s, stride_hs_h, stride_hs_d,
    stride_Y_b, stride_Y_c, stride_Y_i, stride_Y_h, stride_Y_d,
    num_warps=8, num_stages=2
):
    # Grid: (Bsz, Csz, S) — each program handles one i for a given (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)

    # Loop over heads and dims, accumulate over j
    for h in range(H):
        for d in range(D):
            acc = 0.0
            for j in range(S):
                g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
                l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
                hs_off = b * stride_hs_b + c * stride_hs_c + j * stride_hs_s + h * stride_hs_h + d * stride_hs_d

                g_val = tl.load(G_ptr + g_off)
                l_val = tl.load(L_ptr + l_off)
                hs_val = tl.load(hidden_ptr + hs_off)

                # acc += g * l * hidden
                acc += g_val * l_val * hs_val
            # Write Y[b, c, i, h, d]
            y_off = b * stride_Y_b + c * stride_Y_c + i * stride_Y_i + h * stride_Y_h + d * stride_Y_d
            tl.store(Y_ptr + y_off, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Assert shapes for this Triton implementation (specialized to original constants)
        assert hidden_states.shape[-1] == 32, "head_dim D must be 32"
        assert hidden_states.shape[-2] == 32, "num_heads H must be 32"
        assert hidden_states.shape[-3] == 128, "chunk_size S must be 128"
        assert hidden_states.shape[0] == A_cumsum.shape[0], "batch_size mismatch"
        assert hidden_states.shape[1] == A_cumsum.shape[2], "num_chunks mismatch"
        assert A_cumsum.shape[1] == 32, "A_cumsum num_heads must be 32"
        assert A_cumsum.shape[3] == 128, "A_cumsum chunk_size must be 128"
        assert B.shape == (hidden_states.shape[0], hidden_states.shape[1], 128, 8, 32), "B shape mismatch"
        assert C.shape == (hidden_states.shape[0], hidden_states.shape[1], 128, 8, 32), "C shape mismatch"

        Bsz, Csz, S, H, D = hidden_states.shape

        # Ensure contiguous tensors for predictable strides
        A_cumsum = A_cumsum.contiguous()
        B = B.contiguous()
        C = C.contiguous()
        hidden_states = hidden_states.contiguous()

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
            S=128, H=32, num_warps=8, num_stages=2
        )

        # Launch kernel 2: compute G from B and C
        contract_BC_to_G_kernel[(Bsz, Csz)](
            B, C, G,
            Bsz, Csz,
            stride_B_b=B.stride(0), stride_B_c=B.stride(1), stride_B_s=B.stride(2), stride_B_g=B.stride(3), stride_B_k=B.stride(4),
            stride_C_b=C.stride(0), stride_C_c=C.stride(1), stride_C_s=C.stride(2), stride_C_g=C.stride(3), stride_C_k=C.stride(4),
            stride_G_b=G.stride(0), stride_G_c=G.stride(1), stride_G_i=G.stride(2), stride_G_j=G.stride(3), stride_G_h=G.stride(4),
            S=128, G_CONST=8, K=32, num_warps=8, num_stages=2
        )

        # Launch kernel 3: compute Y_diag
        compute_Y_diag_kernel[(Bsz, Csz, S)](
            G, L, hidden_states, Y_diag,
            Bsz, Csz, S, H, D,
            stride_G_b=G.stride(0), stride_G_c=G.stride(1), stride_G_i=G.stride(2), stride_G_j=G.stride(3), stride_G_h=G.stride(4),
            stride_L_b=L.stride(0), stride_L_c=L.stride(1), stride_L_i=L.stride(2), stride_L_j=L.stride(3), stride_L_h=L.stride(4),
            stride_hs_b=hidden_states.stride(0), stride_hs_c=hidden_states.stride(1), stride_hs_s=hidden_states.stride(2),
            stride_hs_h=hidden_states.stride(3), stride_hs_d=hidden_states.stride(4),
            stride_Y_b=Y_diag.stride(0), stride_Y_c=Y_diag.stride(1), stride_Y_i=Y_diag.stride(2),
            stride_Y_h=Y_diag.stride(3), stride_Y_d=Y_diag.stride(4),
            num_warps=8, num_stages=2
        )

        # Return in bfloat16 to match original behavior
        return Y_diag.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
