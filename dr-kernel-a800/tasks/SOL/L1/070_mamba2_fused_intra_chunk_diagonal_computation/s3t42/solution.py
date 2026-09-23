import torch
import triton
import triton.language as tl


@triton.jit
def kernel_A_to_L(
    A_ptr, L_ptr,
    Bsz, Csz,
    stride_A_b, stride_A_h, stride_A_c, stride_A_s,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    S: tl.constexpr,   # chunk_size = 128
    H: tl.constexpr,   # num_heads = 32
    num_warps=8, num_stages=2
):
    # Grid: (Bsz, Csz, H)
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # For each i, build cumsum across j and write L[i, j, h] with lower-triangular mask (i >= j)
    for i in range(S):
        # cumsum vector s_j = sum_{t <= j} A[b, h, c, t]
        s_j = 0.0
        for j in range(S):
            a_off = b * stride_A_b + h * stride_A_h + c * stride_A_c + j * stride_A_s
            a_val = tl.load(A_ptr + a_off)
            s_j += a_val
            # write L[i, j, h] = exp(s_j) if j <= i, else 0
            if j <= i:
                l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
                l_val = tl.exp(s_j)
                tl.store(L_ptr + l_off, l_val)
            else:
                l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
                tl.store(L_ptr + l_off, 0.0)


@triton.jit
def kernel_BC_to_G(
    B_ptr, C_ptr, G_ptr,
    Bsz, Csz,
    stride_B_b, stride_B_c, stride_B_s, stride_B_g, stride_B_k,
    stride_C_b, stride_C_c, stride_C_s, stride_C_g, stride_C_k,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    S: tl.constexpr,          # 128
    G_CONST: tl.constexpr,    # 8
    K: tl.constexpr,          # 32
    num_warps=8, num_stages=2
):
    # Grid: (Bsz, Csz)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Accumulator per (i, j, h)
    for i in range(S):
        for j in range(S):
            for h in range(H):
                acc = 0.0
                for g in range(G_CONST):
                    for k in range(K):
                        b_off = b * stride_B_b + c * stride_B_c + j * stride_B_s + g * stride_B_g + k * stride_B_k
                        c_off = b * stride_C_b + c * stride_C_c + i * stride_C_s + g * stride_C_g + k * stride_C_k
                        b_val = tl.load(B_ptr + b_off)
                        c_val = tl.load(C_ptr + c_off)
                        acc += b_val * c_val
                g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
                tl.store(G_ptr + g_off, acc)


@triton.jit
def kernel_elementwise_mul(
    G_ptr, L_ptr, M_ptr,
    Bsz, Csz,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    stride_M_b, stride_M_c, stride_M_i, stride_M_j, stride_M_h,
    S: tl.constexpr,  # 128
    H: tl.constexpr,  # 32
    num_warps=8, num_stages=2
):
    # Grid: (Bsz, Csz, S, S, H)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
    l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
    g_val = tl.load(G_ptr + g_off)
    l_val = tl.load(L_ptr + l_off)
    m_val = g_val * l_val
    m_off = b * stride_M_b + c * stride_M_c + i * stride_M_i + j * stride_M_j + h * stride_M_h
    tl.store(M_ptr + m_off, m_val)


@triton.jit
def kernel_contract_M_with_hidden(
    M_ptr, hidden_ptr, Y_ptr,
    Bsz, Csz,
    stride_M_b, stride_M_c, stride_M_i, stride_M_j, stride_M_h,
    stride_hs_b, stride_hs_c, stride_hs_j, stride_hs_h, stride_hs_d,
    stride_Y_b, stride_Y_c, stride_Y_i, stride_Y_h, stride_Y_d,
    S: tl.constexpr,  # 128
    H: tl.constexpr,  # 32
    D: tl.constexpr,  # 32
    num_warps=8, num_stages=2
):
    # Grid: (Bsz, Csz, S, H, D)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    d = tl.program_id(4)

    acc = 0.0
    for j in range(S):
        m_off = b * stride_M_b + c * stride_M_c + i * stride_M_i + j * stride_M_j + h * stride_M_h
        hs_off = b * stride_hs_b + c * stride_hs_c + j * stride_hs_j + h * stride_hs_h + d * stride_hs_d
        m_val = tl.load(M_ptr + m_off)
        hs_val = tl.load(hidden_ptr + hs_off)
        acc += m_val * hs_val

    y_off = b * stride_Y_b + c * stride_Y_c + i * stride_Y_i + h * stride_Y_h + d * stride_Y_d
    tl.store(Y_ptr + y_off, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA and contiguous
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "All inputs must be CUDA tensors"
        hidden_states = hidden_states.contiguous()
        A_cumsum = A_cumsum.contiguous()
        B = B.contiguous()
        C = C.contiguous()

        # Shapes (specialized for original constants)
        Bsz, Csz, S, H, D = hidden_states.shape
        assert S == 128 and H == 32 and D == 32, "Specialized Triton implementation expects S=128, H=32, D=32"
        assert A_cumsum.shape == (Bsz, H, Csz, S), f"A_cumsum shape mismatch: expected [B,{H},C,{S}], got {A_cumsum.shape}"
        assert B.shape == (Bsz, Csz, S, 8, 32) and C.shape == (Bsz, Csz, S, 8, 32), "B/C shape mismatch: expected [B,C,S,8,32]"

        # Allocate outputs
        L = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        G = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        M = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        Y_diag = torch.empty((Bsz, Csz, S, H, D), dtype=torch.float32, device=hidden_states.device)

        # Launch kernel 1: A -> L
        kernel_A_to_L[(Bsz, Csz, H)](
            A_cumsum, L,
            Bsz, Csz,
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            S=S, H=H,
            num_warps=8, num_stages=2
        )

        # Launch kernel 2: BC -> G
        kernel_BC_to_G[(Bsz, Csz)](
            B, C, G,
            Bsz, Csz,
            B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            S=S, G_CONST=8, K=32,
            num_warps=8, num_stages=2
        )

        # Launch kernel 3: elementwise M = G * L
        kernel_elementwise_mul[(Bsz, Csz, S, S, H)](
            G, L, M,
            Bsz, Csz,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            S=S, H=H,
            num_warps=8, num_stages=2
        )

        # Launch kernel 4: contract M with hidden to Y_diag
        kernel_contract_M_with_hidden[(Bsz, Csz, S, H, D)](
            M, hidden_states, Y_diag,
            Bsz, Csz,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
            Y_diag.stride(0), Y_diag.stride(1), Y_diag.stride(2), Y_diag.stride(3), Y_diag.stride(4),
            S=S, H=H, D=D,
            num_warps=8, num_stages=2
        )

        # Return in bfloat16 as the original code
        return Y_diag.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
