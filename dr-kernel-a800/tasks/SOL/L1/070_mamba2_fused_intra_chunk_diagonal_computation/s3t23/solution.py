import torch
import triton
import triton.language as tl


@triton.jit
def compute_L_exp_cumsum_kernel(
    A_ptr, L_ptr,
    Bsz, Csz, S, H,
    stride_A_b, stride_A_h, stride_A_c, stride_A_s,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    num_warps=8, num_stages=2
):
    # Grid: (Bsz, Csz*S) -> each program handles (b, c, i)
    pid_b = tl.program_id(0)
    pid_cSi = tl.program_id(1)
    c = pid_cSi // S
    i = pid_cSi % S

    # For each target j, compute s_j = sum over h of A[b, h, c, j], then write L[i, j, h] = exp(s_j) if i >= j else 0
    for j in range(S):
        s_j = 0.0
        for hh in range(H):
            a_off = pid_b * stride_A_b + hh * stride_A_h + c * stride_A_c + j * stride_A_s
            a_val = tl.load(A_ptr + a_off)
            s_j += a_val
        l_val = tl.exp(s_j)
        # Store for all heads h
        for hh in range(H):
            l_off = pid_b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + hh * stride_L_h
            tl.store(L_ptr + l_off, l_val)
        # For j > i, we should store 0.0 (triangular mask)
        if j > i:
            for hh in range(H):
                l_off = pid_b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + hh * stride_L_h
                tl.store(L_ptr + l_off, 0.0)


@triton.jit
def contract_BC_to_G_kernel(
    B_ptr, C_ptr, G_ptr,
    Bsz, Csz, S, H,
    stride_B_b, stride_B_c, stride_B_s, stride_B_g, stride_B_k,
    stride_C_b, stride_C_c, stride_C_s, stride_C_g, stride_C_k,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    G_CONST: tl.constexpr,  # 8
    K: tl.constexpr,        # 32
    num_warps=8, num_stages=2
):
    # Grid: (Bsz, Csz*S) -> each program handles (b, c, i)
    pid_b = tl.program_id(0)
    pid_cSi = tl.program_id(1)
    c = pid_cSi // S
    i = pid_cSi % S

    # Accumulator for G[i, j, h] across j
    for j in range(S):
        acc = 0.0
        # Sum over groups g and state k
        for g in range(G_CONST):
            for k in range(K):
                b_off = pid_b * stride_B_b + c * stride_B_c + j * stride_B_s + g * stride_B_g + k * stride_B_k
                c_off = pid_b * stride_C_b + c * stride_C_c + i * stride_C_s + g * stride_C_g + k * stride_C_k
                b_val = tl.load(B_ptr + b_off)
                c_val = tl.load(C_ptr + c_off)
                acc += b_val * c_val
        # Write G[i, j, h] for all heads h
        for hh in range(H):
            g_off = pid_b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + hh * stride_G_h
            tl.store(G_ptr + g_off, acc)


@triton.jit
def contract_M_with_hidden_kernel(
    G_ptr, L_ptr, hidden_ptr, Y_ptr,
    Bsz, Csz, S, H, D,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    stride_H_b, stride_H_c, stride_H_s, stride_H_h, stride_H_d,
    stride_Y_b, stride_Y_c, stride_Y_i, stride_Y_h, stride_Y_d,
    num_warps=8, num_stages=2
):
    # Grid: (Bsz, Csz, S, H) -> each program handles (b, c, i, h)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    acc = 0.0
    # Loop over j and D
    for j in range(S):
        # Load G[i, j, h]
        g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
        g_val = tl.load(G_ptr + g_off)
        # Load L[i, j, h]
        l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
        l_val = tl.load(L_ptr + l_off)
        m_val = g_val * l_val
        # Load hidden[b, c, j, h, d] and accumulate
        for d in range(D):
            h_off = b * stride_H_b + c * stride_H_c + j * stride_H_s + h * stride_H_h + d * stride_H_d
            h_val = tl.load(hidden_ptr + h_off)
            acc += m_val * h_val

    # Store Y[b, c, i, h, 0] (D is 32 in original; we store into first dim and rely on contiguous layout if D=32)
    y_off = b * stride_Y_b + c * stride_Y_c + i * stride_Y_i + h * stride_Y_h + 0 * stride_Y_d
    tl.store(Y_ptr + y_off, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Inputs must be CUDA tensors"
        hidden_states = hidden_states.contiguous()
        A_cumsum = A_cumsum.contiguous()
        B = B.contiguous()
        C = C.contiguous()

        # Extract shapes
        Bsz, Csz, S, H, D = hidden_states.shape
        assert A_cumsum.shape == (Bsz, H, Csz, S), f"Expected A_cumsum shape [B,H,C,S], got {A_cumsum.shape}"
        # Constants for original model
        G_CONST = 8
        K = 32

        # Allocate outputs
        L = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        G = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        # Y_diag will be [B, C, S, H, D]; we compute into float32 and return bfloat16
        Y = torch.empty((Bsz, Csz, S, H, 1), dtype=torch.float32, device=hidden_states.device)  # we will expand by copying

        # Launch kernel 1: compute L
        compute_L_exp_cumsum_kernel[(Bsz, Csz * S)](
            A_cumsum, L,
            Bsz, Csz, S, H,
            stride_A_b=A_cumsum.stride(0), stride_A_h=A_cumsum.stride(1), stride_A_c=A_cumsum.stride(2), stride_A_s=A_cumsum.stride(3),
            stride_L_b=L.stride(0), stride_L_c=L.stride(1), stride_L_i=L.stride(2), stride_L_j=L.stride(3), stride_L_h=L.stride(4),
            num_warps=8, num_stages=2
        )

        # Launch kernel 2: compute G
        contract_BC_to_G_kernel[(Bsz, Csz * S)](
            B, C, G,
            Bsz, Csz, S, H,
            stride_B_b=B.stride(0), stride_B_c=B.stride(1), stride_B_s=B.stride(2), stride_B_g=B.stride(3), stride_B_k=B.stride(4),
            stride_C_b=C.stride(0), stride_C_c=C.stride(1), stride_C_s=C.stride(2), stride_C_g=C.stride(3), stride_C_k=C.stride(4),
            stride_G_b=G.stride(0), stride_G_c=G.stride(1), stride_G_i=G.stride(2), stride_G_j=G.stride(3), stride_G_h=G.stride(4),
            G_CONST=8, K=32, num_warps=8, num_stages=2
        )

        # Launch kernel 3: compute Y_diag by contracting M=G*L with hidden
        # We need Y with D dimension; since D=32 in original, we can compute into Y[..., 0] and expand, but simpler: compute into a full [B,C,S,H,D] tensor
        Y_full = torch.empty((Bsz, Csz, S, H, D), dtype=torch.float32, device=hidden_states.device)

        contract_M_with_hidden_kernel[(Bsz, Csz, S, H)](
            G, L, hidden_states, Y_full,
            Bsz, Csz, S, H, D,
            stride_G_b=G.stride(0), stride_G_c=G.stride(1), stride_G_i=G.stride(2), stride_G_j=G.stride(3), stride_G_h=G.stride(4),
            stride_L_b=L.stride(0), stride_L_c=L.stride(1), stride_L_i=L.stride(2), stride_L_j=L.stride(3), stride_L_h=L.stride(4),
            stride_H_b=hidden_states.stride(0), stride_H_c=hidden_states.stride(1), stride_H_s=hidden_states.stride(2), stride_H_h=hidden_states.stride(3), stride_H_d=hidden_states.stride(4),
            stride_Y_b=Y_full.stride(0), stride_Y_c=Y_full.stride(1), stride_Y_i=Y_full.stride(2), stride_Y_h=Y_full.stride(3), stride_Y_d=Y_full.stride(4),
            num_warps=8, num_stages=2
        )

        # Return in bfloat16 to match original output dtype
        return Y_full.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
