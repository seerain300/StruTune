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

    # For each i, build cumsum across j and write L[i, j, h] with lower-triangular mask (i >= j)
    for i in range(S):
        for j in range(S):
            s_j = 0.0
            for hh in range(H):
                a_off = b * stride_A_b + hh * stride_A_h + c * stride_A_c + j * stride_A_s
                a_val = tl.load(A_ptr + a_off)
                s_j += a_val
            # Now set L[i, j, h] = exp(s_j) for all h (only when j <= i)
            for hh in range(H):
                l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + hh * stride_L_h
                l_val = tl.exp(s_j)
                # Store only if j <= i (triangular mask)
                if j <= i:
                    tl.store(L_ptr + l_off, l_val)
                # else leave as 0 by not storing (L is initialized to zeros)


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

    # Accumulate G[i, j, h] = sum_g sum_k C[b, c, i, g, k] * B[b, c, j, g, k]
    for i in range(S):
        for j in range(S):
            acc = 0.0
            for hh in range(H):  # H=32, we sum contributions for each head h
                for g in range(G_CONST):
                    for k in range(K):
                        b_off = b * stride_B_b + c * stride_B_c + j * stride_B_s + g * stride_B_g + k * stride_B_k
                        c_off = b * stride_C_b + c * stride_C_c + i * stride_C_s + g * stride_C_g + k * stride_C_k
                        b_val = tl.load(B_ptr + b_off)
                        c_val = tl.load(C_ptr + c_off)
                        acc += b_val * c_val
                # Store acc into G[i, j, h]
                g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + hh * stride_G_h
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
    # Grid: (Bsz, Csz, S, H) — one program per (b, c, i, h), and loop over d and j inside
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    acc_vec = tl.zeros((D,), dtype=tl.float32)

    # Loop over j from 0..S-1
    for j in range(S):
        g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
        l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h

        G_val = tl.load(G_ptr + g_off)
        L_val = tl.load(L_ptr + l_off)
        prod = G_val * L_val

        # Loop over d from 0..D-1
        for d in range(D):
            hs_off = b * stride_hs_b + c * stride_hs_c + j * stride_hs_s + h * stride_hs_h + d * stride_hs_d
            hs_val = tl.load(hidden_ptr + hs_off)
            acc_vec[d] += prod * hs_val

    # Store acc_vec to Y[b, c, i, h, :]
    for d in range(D):
        y_off = b * stride_Y_b + c * stride_Y_c + i * stride_Y_i + h * stride_Y_h + d * stride_Y_d
        tl.store(Y_ptr + y_off, acc_vec[d])


def run_triton_path(Bsz: int, Csz: int, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    # Assume fixed constants as per original model: S=128, H=32, D=32, G_CONST=8, K=32
    assert hidden_states.shape == (Bsz, Csz, 128, 32, 32), "hidden_states must be [B, C, S=128, H=32, D=32]"
    assert A_cumsum.shape == (Bsz, 32, Csz, 128), "A_cumsum must be [B, H=32, C, S=128]"
    assert B.shape == (Bsz, Csz, 128, 8, 32), "B must be [B, C, S=128, G_CONST=8, K=32]"
    assert C.shape == (Bsz, Csz, 128, 8, 32), "C must be [B, C, S=128, G_CONST=8, K=32]"

    # Ensure CUDA and contiguous
    assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "All inputs must be CUDA tensors"
    hidden_states = hidden_states.contiguous()
    A_cumsum = A_cumsum.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # Allocate outputs
    L = torch.empty((Bsz, Csz, 128, 128, 32), dtype=torch.float32, device=hidden_states.device)
    G = torch.empty((Bsz, Csz, 128, 128, 32), dtype=torch.float32, device=hidden_states.device)
    Y_diag = torch.empty((Bsz, Csz, 128, 32, 32), dtype=torch.float32, device=hidden_states.device)

    # Launch Triton kernel to compute L
    compute_L_exp_cumsum_kernel[(Bsz, Csz)](
        A_cumsum, L,
        Bsz, Csz,
        A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
        L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
        S=128, H=32,
        num_warps=8, num_stages=2
    )

    # Launch Triton kernel to compute G from B and C
    contract_BC_to_G_kernel[(Bsz, Csz)](
        B, C, G,
        Bsz, Csz,
        B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
        G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
        S=128, G_CONST=8, K=32,
        num_warps=8, num_stages=2
    )

    # Launch Triton kernel to compute Y_diag by contracting M = G * L with hidden_states
    compute_Y_diag_kernel[(Bsz, Csz, 128, 32)](
        G, L, hidden_states, Y_diag,
        Bsz, Csz, 128, 32, 32,
        G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
        L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
        Y_diag.stride(0), Y_diag.stride(1), Y_diag.stride(2), Y_diag.stride(3), Y_diag.stride(4),
        num_warps=8, num_stages=2
    )

    return Y_diag.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # All computation in Triton
        return run_triton_path(hidden_states.shape[0], hidden_states.shape[1], A_cumsum, B, C, hidden_states)


def run(*args):
    return ModelNew()(*args)
