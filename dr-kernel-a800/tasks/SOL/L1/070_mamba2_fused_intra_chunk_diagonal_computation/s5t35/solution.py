import torch
import triton
import triton.language as tl


@triton.jit
def build_L_kernel(
    A_ptr,  # [B, C, L, H], float32
    L_ptr,  # [B, C, L, L, H], float32
    B: tl.constexpr, C: tl.constexpr, L: tl.constexpr, H: tl.constexpr,
    stride_A_b, stride_A_c, stride_A_l, stride_A_h,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
):
    # 5D grid: (b, c, i, j, h)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    if (i >= L) or (j >= L):
        return

    # Compute total = sum over k=0..L-1 of A[b, c, k, h]
    total = 0.0
    for k in range(0, L):
        a_ptr = A_ptr + b * stride_A_b + c * stride_A_c + k * stride_A_l + h * stride_A_h
        a_val = tl.load(a_ptr)
        total += a_val

    # Lower-triangular mask with diagonal=-1: if j <= i, set L[i, j] = exp(total), else 0
    if j <= i:
        l_ptr = L_ptr + b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
        tl.store(l_ptr, tl.exp(total))
    else:
        l_ptr = L_ptr + b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
        tl.store(l_ptr, 0.0)


@triton.jit
def expand_groups_repeat_interleave(
    T_src_ptr,  # [B, C, L, G, S], float32 (T_src can be B or C)
    T_dst_ptr,  # [B, C, L, H, S], float32
    B: tl.constexpr, C: tl.constexpr, L: tl.constexpr, G: tl.constexpr, S: tl.constexpr,
    stride_src_b, stride_src_c, stride_src_l, stride_src_g, stride_src_s,
    stride_dst_b, stride_dst_c, stride_dst_l, stride_dst_h, stride_dst_s,
):
    # 5D grid: (b, c, i, h, s)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    s = tl.program_id(4)

    if (b >= B) or (c >= C) or (i >= L) or (h >= H) or (s >= S):
        return

    # Map head to original group index g = h // G; expanded index is h*G + g == h
    g = h // G
    src_ptr = T_src_ptr + b * stride_src_b + c * stride_src_c + i * stride_src_l + g * stride_src_g + s * stride_src_s
    val = tl.load(src_ptr)
    dst_ptr = T_dst_ptr + b * stride_dst_b + c * stride_dst_c + i * stride_dst_l + h * stride_dst_h + s * stride_dst_s
    tl.store(dst_ptr, val)


@triton.jit
def compute_G_kernel(
    C_exp_ptr,  # [B, C, L, H, S], float32
    B_exp_ptr,  # [B, C, L, H, S], float32
    G_ptr,      # [B, C, L, L, H], float32
    B: tl.constexpr, C: tl.constexpr, L: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    stride_Cb, stride_Cc, stride_Ci, stride_Ch, stride_Cs,
    stride_Bb, stride_Bc, stride_Bj, stride_Bh, stride_Bs,
    stride_Gb, stride_Gc, stride_Gi, stride_Gj, stride_Gh,
):
    # 5D grid: (b, c, i, j, h)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    if (b >= B) or (c >= C) or (i >= L) or (j >= L) or (h >= H):
        return

    acc = 0.0
    for s in range(0, S):
        c_ptr = C_exp_ptr + b * stride_Cb + c * stride_Cc + i * stride_Ci + h * stride_Ch + s * stride_Cs
        b_ptr = B_exp_ptr + b * stride_Bb + c * stride_Bc + j * stride_Bj + h * stride_Bh + s * stride_Bs
        c_val = tl.load(c_ptr)
        b_val = tl.load(b_ptr)
        acc += c_val * b_val

    g_ptr = G_ptr + b * stride_Gb + c * stride_Gc + i * stride_Gi + j * stride_Gj + h * stride_Gh
    tl.store(g_ptr, acc)


@triton.jit
def compute_Y_diag_kernel(
    M_ptr,          # [B, C, L, L, H], float32
    hidden_ptr,     # [B, C, L, H, D], float32
    Y_ptr,          # [B, C, L, H, D], float32
    B: tl.constexpr, C: tl.constexpr, L: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    stride_M_b, stride_M_c, stride_M_i, stride_M_j, stride_M_h,
    stride_h_b, stride_h_c, stride_h_j, stride_h_h, stride_h_d,
    stride_Y_b, stride_Y_c, stride_Y_i, stride_Y_h, stride_Y_d,
):
    # 5D grid: (b, c, i, h, d)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    d = tl.program_id(4)

    if (b >= B) or (c >= C) or (i >= L) or (h >= H) or (d >= D):
        return

    acc = 0.0
    for j in range(0, L):
        m_ptr = M_ptr + b * stride_M_b + c * stride_M_c + i * stride_M_i + j * stride_M_j + h * stride_M_h
        h_ptr = hidden_ptr + b * stride_h_b + c * stride_h_c + j * stride_h_j + h * stride_h_h + d * stride_h_d
        m_val = tl.load(m_ptr)
        h_val = tl.load(h_ptr)
        acc += m_val * h_val

    y_ptr = Y_ptr + b * stride_Y_b + c * stride_Y_c + i * stride_Y_i + h * stride_Y_h + d * stride_Y_d
    tl.store(y_ptr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants fixed as in original code
        self.NUM_HEADS = 32
        self.N_GROUPS = 8
        self.GROUP_EXPAND = 4  # NUM_HEADS // N_GROUPS

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Extract shapes (dynamic from input)
        batch_size, num_chunks, chunk_size, num_heads, head_dim = hidden_states.shape

        # Ensure inputs on same device and dtype handling
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Convert to float32 for compute
        hidden_f32 = hidden_states.to(torch.float32)
        A_f32 = A_cumsum.to(torch.float32)  # [B, C, L, H]
        B_f32 = B.to(torch.float32)
        C_f32 = C.to(torch.float32)

        # 1) Build L matrix [B, C, L, L, H] using Triton
        L = torch.empty((batch_size, num_chunks, chunk_size, chunk_size, num_heads), dtype=torch.float32, device=device)
        grid_L = (batch_size, num_chunks, chunk_size, chunk_size, num_heads)
        build_L_kernel[grid_L](
            A_f32, L,
            batch_size, num_chunks, chunk_size, num_heads,
            A_f32.stride(0), A_f32.stride(1), A_f32.stride(2), A_f32.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
        )

        # 2) Expand B and C from groups to heads using Triton (N_GROUPS=8 -> H=32, GROUP_EXPAND=4)
        S_size = C_f32.shape[4]  # state_size
        B_exp = torch.empty((batch_size, num_chunks, chunk_size, num_heads, S_size), dtype=torch.float32, device=device)
        C_exp = torch.empty((batch_size, num_chunks, chunk_size, num_heads, S_size), dtype=torch.float32, device=device)

        grid_expand = (batch_size, num_chunks, chunk_size, num_heads, S_size)
        expand_groups_repeat_interleave[grid_expand](
            B_f32, B_exp,
            batch_size, num_chunks, chunk_size, self.N_GROUPS, S_size,
            B_f32.stride(0), B_f32.stride(1), B_f32.stride(2), B_f32.stride(3), B_f32.stride(4),
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
        )
        expand_groups_repeat_interleave[grid_expand](
            C_f32, C_exp,
            batch_size, num_chunks, chunk_size, self.N_GROUPS, S_size,
            C_f32.stride(0), C_f32.stride(1), C_f32.stride(2), C_f32.stride(3), C_f32.stride(4),
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
        )

        # 3) Compute G = sum_s C_exp * B_exp in Triton: G[b, c, i, j, h] = sum_s C[b,c,i,h,s] * B[b,c,j,h,s]
        G = torch.empty((batch_size, num_chunks, chunk_size, chunk_size, num_heads), dtype=torch.float32, device=device)
        grid_G = (batch_size, num_chunks, chunk_size, chunk_size, num_heads)
        compute_G_kernel[grid_G](
            C_exp, B_exp, G,
            batch_size, num_chunks, chunk_size, num_heads, S_size,
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
        )

        # 4) Apply causal mask: M = G * L elementwise (PyTorch for simplicity)
        M = G * L  # L is same shape as G: [B, C, L, L, H]

        # 5) Compute Y_diag by contraction over j: Y[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden[b, c, j, h, d] in Triton
        Y = torch.empty((batch_size, num_chunks, chunk_size, num_heads, head_dim), dtype=torch.float32, device=device)
        grid_Y = (batch_size, num_chunks, chunk_size, num_heads, head_dim)
        compute_Y_diag_kernel[grid_Y](
            M, hidden_f32, Y,
            batch_size, num_chunks, chunk_size, num_heads, head_dim,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden_f32.stride(0), hidden_f32.stride(1), hidden_f32.stride(2), hidden_f32.stride(3), hidden_f32.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
        )

        # Return in bfloat16 to match original behavior
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
