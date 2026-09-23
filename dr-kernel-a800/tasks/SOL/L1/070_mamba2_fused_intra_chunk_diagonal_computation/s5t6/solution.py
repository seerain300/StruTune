import torch
import triton
import triton.language as tl

# Triton kernel: build L = exp(cumsum(A_masked)) where A_masked is A_cumsum with lower-triangular mask
@triton.jit
def build_L_kernel(
    A_ptr,         # *float32, [B, C, L, H]
    L_ptr,         # *float32, [B, H, C, L, L]
    B_size: tl.constexpr,  # int
    C_size: tl.constexpr,  # int
    L_len: tl.constexpr,   # int
    H_size: tl.constexpr,  # int
):
    # One program per (b, h)
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Loop over c (num_chunks); we store L for each c
    for c in range(0, C_size):
        # For each row i, compute cumsum along internal dim L, then exp
        # We'll build L[i, j] for j in [0..L_len-1]:
        #   if i >= j: L[i, j] = exp(sum_{k=0..i} A[b, c, k, h]); else 0
        for i in range(0, L_len):
            row_sum = 0.0
            # Compute cumsum of A[b, c, :, h] up to i
            for k in range(0, L_len):
                a_addr = b * (C_size * L_len * H_size) + c * (L_len * H_size) + k * H_size + h
                a_val = tl.load(A_ptr + a_addr)
                row_sum += a_val
            # Now set lower-triangular entries
            for j in range(0, L_len):
                if j <= i:
                    l_addr = b * (H_size * C_size * L_len * L_len) + h * (C_size * L_len * L_len) + c * (L_len * L_len) + i * L_len + j
                    tl.store(L_ptr + l_addr, tl.exp(row_sum))
                else:
                    l_addr = b * (H_size * C_size * L_len * L_len) + h * (C_size * L_len * L_len) + c * (L_len * L_len) + i * L_len + j
                    tl.store(L_ptr + l_addr, 0.0)


# Triton kernel: compute G[b, c, i, j, h] = sum_s C_exp[b, c, i, h, s] * B_exp[b, c, j, h, s]
@triton.jit
def g_outer_kernel(
    B_exp_ptr,    # *float32, [B, C, L, H, S]
    C_exp_ptr,    # *float32, [B, C, L, H, S]
    G_ptr,        # *float32, [B, C, L, L, H]
    B_size: tl.constexpr,  # int
    C_size: tl.constexpr,  # int
    L_len: tl.constexpr,   # int
    H_size: tl.constexpr,  # int
    S_size: tl.constexpr,  # int (state_size)
):
    # 3D grid: (pid0 over B*C, pid1 over L*L, pid2 over H)
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    b = pid0 // C_size
    c = pid0 % C_size
    i = pid1 // L_len
    j = pid1 % L_len
    h = pid2

    g_val = 0.0
    # Loop over state dimension S
    for s in range(0, S_size):
        B_addr = b * (C_size * L_len * H_size * S_size) + c * (L_len * H_size * S_size) + j * (H_size * S_size) + h * S_size + s
        C_addr = b * (C_size * L_len * H_size * S_size) + c * (L_len * H_size * S_size) + i * (H_size * S_size) + h * S_size + s
        B_val = tl.load(B_exp_ptr + B_addr)
        C_val = tl.load(C_exp_ptr + C_addr)
        g_val += C_val * B_val

    # Store G[b, c, i, j, h]
    G_stride_b = C_size * L_len * L_len * H_size
    G_stride_c = L_len * L_len * H_size
    G_stride_i = L_len * H_size
    G_stride_j = H_size
    G_stride_h = 1
    G_addr = b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + h * G_stride_h
    tl.store(G_ptr + G_addr, g_val)


# Triton kernel: M = G * L_permuted where L_permuted is [B, C, H, L, L]
@triton.jit
def multiply_mask_kernel(
    G_ptr,         # *float32, [B, C, L, L, H]
    L_perm_ptr,    # *float32, [B, C, H, L, L]
    M_ptr,         # *float32, [B, C, L, L, H]
    B_size: tl.constexpr,  # int
    C_size: tl.constexpr,  # int
    L_len: tl.constexpr,   # int
    H_size: tl.constexpr,  # int
):
    # 3D grid over (B, C, H)
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # Vectorize over i, j in [0..L_len-1]
    i_vec = tl.arange(0, L_len)
    j_vec = tl.arange(0, L_len)

    # G[b, c, i, j, h] as [L_len, L_len]
    G_offsets = i_vec[:, None] * (L_len * H_size) + j_vec[None, :] * H_size + h
    G_addrs = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + G_offsets
    G_mat = tl.load(G_ptr + G_addrs)

    # L_perm[b, c, h, i, j] as [L_len, L_len]
    L_perm_addrs = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + i_vec[:, None] * (L_len * H_size) + j_vec[None, :] * H_size + h
    L_perm_mat = tl.load(L_perm_ptr + L_perm_addrs)

    M_mat = G_mat * L_perm_mat

    # Store M[b, c, i, j, h]
    M_addrs = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + i_vec[:, None] * (L_len * H_size) + j_vec[None, :] * H_size + h
    tl.store(M_ptr + M_addrs, M_mat)


# Triton kernel: compute Y_diag[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden_states[b, c, j, h, d]
@triton.jit
def y_diag_reduce_kernel(
    M_ptr,         # *float32, [B, C, L, L, H]
    HS_ptr,        # *float32, [B, C, L, H, D]
    Y_ptr,         # *float32, [B, C, L, H, D]
    B_size: tl.constexpr,  # int
    C_size: tl.constexpr,  # int
    L_len: tl.constexpr,   # int
    H_size: tl.constexpr,  # int
    D_size: tl.constexpr,  # int
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    d_offsets = tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for j in range(0, L_len):
        M_addr = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + i * (L_len * H_size) + j * H_size + h
        m_val = tl.load(M_ptr + M_addr)
        HS_base = b * (C_size * L_len * H_size * D_size) + c * (L_len * H_size * D_size) + j * (H_size * D_size) + h * D_size
        HS_addr = HS_base + d_offsets
        hs_vec = tl.load(HS_ptr + HS_addr, mask=d_offsets < D_size)
        acc += m_val * hs_vec

    Y_base = b * (C_size * L_len * H_size * D_size) + c * (L_len * H_size * D_size) + i * (H_size * D_size) + h * D_size
    tl.store(Y_ptr + Y_base + d_offsets, acc, mask=d_offsets < D_size)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, C, L, H, D]
        A_cumsum:      [B, C, L, H]
        B:             [B, C, L, groups, S]
        C:             [B, C, L, groups, S]
        Returns:       [B, C, L, H, D] in bfloat16
        """
        device = hidden_states.device
        B_size, C_size, L_len, H_size, D_size = hidden_states.shape
        # We don't have explicit S_size in inputs; in the original, it's B.shape[4]. Use B's last dim.
        # The reference code does not pass S separately; we assume B and C share the same S.
        # For robustness, fetch S_size from B tensors; but since B is passed, we can use B.shape[4].
        # However, to be safe, we infer S_size from B's last dim. The original PyTorch code expects B and C to have same S.
        # Here, we assume B and C have last dim S. If not provided, we need to infer. The original function signature doesn't pass S.
        # To proceed, we require B and C to have last dim S; in typical usage, S is known. If S is not known, this code cannot


def run(*args):
    return ModelNew()(*args)
