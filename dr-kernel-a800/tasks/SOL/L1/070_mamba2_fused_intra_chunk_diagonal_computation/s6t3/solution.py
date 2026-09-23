import torch
import triton
import triton.language as tl


@triton.jit
def tril_mask_kernel(M: tl.pointer_type(tl.int8), S: tl.constexpr):
    # Produce a lower-triangular mask with diagonal=-1: j <= i -> True (1), else False (0)
    i = tl.program_id(0)
    j = tl.program_id(1)
    if j <= i:
        M[i * S + j] = 1
    else:
        M[i * S + j] = 0


@triton.jit
def exp_cumsum_tril_kernel(
    A_ptr,  # [B, H, N, S, S] as linearized (contiguous expand)
    L_ptr,  # [B, H, N, S, S] as linearized
    B_size: tl.constexpr, H_size: tl.constexpr, N_size: tl.constexpr, S_size: tl.constexpr,
    A_stride_b, A_stride_h, A_stride_n, A_stride_s1, A_stride_s2,
    Out_stride_b, Out_stride_h, Out_stride_n, Out_stride_s1, Out_stride_s2,
):
    # Each program handles one (b, h, n, i)
    b = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)
    i = tl.program_id(3)  # iterate over S dimension

    base = b * A_stride_b + h * A_stride_h + n * A_stride_n
    prefix = 0.0
    for j in range(0, S_size):
        offset = base + j * A_stride_s2
        a = tl.load(A_ptr + offset)
        add = 1.0 if (j <= i) else 0.0
        prefix += a * add
        out_offset = b * Out_stride_b + h * Out_stride_h + n * Out_stride_n + i * Out_stride_s1 + j * Out_stride_s2
        val = tl.exp(prefix) if (j <= i) else 0.0
        tl.store(L_ptr + out_offset, val)


@triton.jit
def g_contract_kernel(
    C_ptr,  # [B, N, S, H, D]
    B_ptr,  # [B, N, S, H, D]
    G_ptr,  # [B, N, S, S, H] (float32)
    B_size: tl.constexpr, N_size: tl.constexpr, S_size: tl.constexpr, H_size: tl.constexpr, D_size: tl.constexpr,
    C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
    B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)  # source
    j = tl.program_id(3)  # target
    h = tl.program_id(4)

    acc = 0.0
    for d0 in range(0, D_size, BLOCK_D):
        d_range = d0 + tl.arange(0, BLOCK_D)
        mask = d_range < D_size
        c_off = b * C_stride_b + n * C_stride_n + i * C_stride_s + h * C_stride_h + d_range * C_stride_d
        c = tl.load(C_ptr + c_off, mask=mask, other=0.0)
        b_off = b * B_stride_b + n * B_stride_n + j * B_stride_s + h * B_stride_h + d_range * B_stride_d
        b2 = tl.load(B_ptr + b_off, mask=mask, other=0.0)
        acc += tl.sum(c * b2, axis=0)
    g_off = b * G_stride_b + n * G_stride_n + i * G_stride_s1 + j * G_stride_s2 + h * G_stride_h
    tl.store(G_ptr + g_off, acc)


@triton.jit
def m_mul_kernel(
    G_ptr,  # [B, N, S, S, H]
    L_ptr,  # [B, H, N, S, S] (we'll index as per our strides; but since we expand A, L is in [B, N, S, S, H]?)
    # Note: M shapes are [B, N, S, S, H] = G; L must be [B, N, S, S, H] so we can multiply elementwise.
    # We pass L as [B, H, N, S, S] and remap indexing to match G's (b,n) ordering in the same way as G kernel did.
    B_size: tl.constexpr, N_size: tl.constexpr, S_size: tl.constexpr, H_size: tl.constexpr,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
):
    # Grid over (b, n, i, j, h)
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    g_off = b * G_stride_b + n * G_stride_n + i * G_stride_s1 + j * G_stride_s2 + h * G_stride_h
    g_val = tl.load(G_ptr + g_off)

    # For L: original passed as [B, H, N, S, S]; we need to index it as [B, N, S, S, H] to match M.
    # Since Triton does not support arbitrary dim reordering for loads, we assume here that L was produced by
    # our exp_cumsum_tril_kernel which writes L in the desired [B, H, N, S, S] order; we must permute in host
    # before passing. To strictly keep Triton-only and avoid host-side permute, we will instead produce L in
    # the correct layout at the time of writing L: our exp_cumsum_tril_kernel writes into Out strides (b,h,n,s1,s2)
    # where s1 is i and s2 is j, and we pass L_ptr accordingly. Thus, L is already [B, H, N, S, S]. The M kernel
    # will load L with indices (b, h, n, i, j), which matches G's (b,n,i,j,h). If G and L are both [B, H, N, S, S],
    # we can simply load L at (b, h, n, i, j) and multiply.

    # Therefore, we need L_ptr to represent [B, H, N, S, S]. Our kernel expects L_stride_b, L_stride_h, L_stride_n, etc.
    # The host will pass L strides that correspond to [B, H, N, S, S].
    l_off = b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_s1 + j * L_stride_s2
    l_val = tl.load(L_ptr + l_off)

    m_val = g_val * l_val
    m_off = b * M_stride_b + n * M_stride_n + i * M_stride_s1 + j * M_stride_s2 + h * M_stride_h
    tl.store(M_ptr + m_off, m_val)


@triton.jit
def y_diag_reduce_kernel(
    M_ptr,       # [B, N, S, S, H] float32
    hidden_ptr,  # [B, N, S, H, D] float32
    Y_ptr,       # [B, N, S, H] float32
    B_size: tl.constexpr, N_size: tl.constexpr, S_size: tl.constexpr, H_size: tl.constexpr, D_size: tl.constexpr,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
    hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
    Y_stride_b, Y_stride_n, Y_stride_s, Y_stride_h,
    BLOCK_J: tl.constexpr,
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    acc = 0.0
    for j0 in range(0, S_size, BLOCK_J):
        j_range = j0 + tl.arange(0, BLOCK_J)
        mask = j_range < S_size
        g_off = b * M_stride_b + n * M_stride_n + i * M_stride_s1 + j_range * M_stride_s2 + h * M_stride_h
        g_vec = tl.load(M_ptr + g_off, mask=mask, other=0.0)  # [BLOCK_J]

        hid_off_base = b * hidden_stride_b + n * hidden_stride_n + h * hidden_stride_h
        # We need to load hidden[b, n, j, h, :] and multiply-accumulate with g_vec[j]
        dot_tile = tl.zeros((BLOCK_J,), dtype=tl.float32)
        for d0 in range(0, D_size, 16):
            d_range = d0 + tl.arange(0, 16)
            mask_d = d_range < D_size
            col_off = hid_off_base + j_range * hidden_stride_s + d_range[None, :] * hidden_stride_d
            # We must broadcast j_range over d_range. Triton supports per-d loop:
            # Iterate d per element
            for d in range(0, 16):
                d_curr = d0 + d
                if d_curr >= D_size:
                    break
                col_off_curr = hid_off_base + j_range * hidden_stride_s + d_curr * hidden_stride_d
                col = tl.load(hidden_ptr + col_off_curr, mask=mask, other=0.0)  # [BLOCK_J]
                dot_tile += g_vec * col
        acc += tl.sum(dot_tile, axis=0)

    y_off = b * Y_stride_b + n * Y_stride_n + i * Y_stride_s + h * Y_stride_h
    tl.store(Y_ptr + y_off, acc)


class ModelNew(torch.nn.Module):
    NUM_HEADS = 32
    N_GROUPS = 8

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor):
        # Validate shapes
        B_size, N_size, S_size, H_size, D_size = hidden_states.shape
        assert A_cumsum.shape == (B_size, H_size, N_size, S_size), "A_cumsum must have shape [B, H, N, S]"
        assert B.shape == (B_size, N_size, S_size, H_size, D_size), "B must have shape [B, N, S, H, D]"
        assert C.shape == (B_size, N_size, S_size, H_size, D_size), "C must have shape [B, N, S, H, D]"
        assert H_size == self.NUM_HEADS, "num_heads must be 32"
        assert device.type == 'cuda', "ModelNew requires CUDA device for Triton kernels"

        # 1) Triton: Lower-triangular mask M_lower of shape [S, S] with diagonal=-1
        M_lower = torch.empty((S_size, S_size), dtype=torch.int8, device=hidden_states.device)
        grid_mask = (S_size, S_size)
        tril_mask_kernel[grid_mask](M_lower, S_size, num_warps=1, num_stages=1)

        # 2) Triton: L = exp(cumsum(masked A)) along S dimension for each (b, h, n, i)
        # Expand A_cumsum to [B, H, N, S, S]
        A_expanded_contig = A_cumsum.unsqueeze(-1).expand(B_size, H_size, N_size, S_size, S_size).to(torch.float32).contiguous()
        L = torch.empty((B_size, H_size, N_size, S_size, S_size), dtype=torch.float32, device=hidden_states.device)

        A_stride_b, A_stride_h, A_stride_n, A_stride_s1, A_stride_s2 = A_expanded_contig.stride()
        Out_stride_b, Out_stride_h, Out_stride_n, Out_stride_s1, Out_stride_s2 = L.stride()
        grid = (B_size, H_size, N_size, S_size)
        exp_cumsum_tril_kernel[grid](
            A_expanded_contig, L,
            B_size, H_size, N_size, S_size,
            A_stride_b, A_stride_h, A_stride_n, A_stride_s1, A_stride_s2,
            Out_stride_b, Out_stride_h, Out_stride_n, Out_stride_s1, Out_stride_s2,
            num_warps=1, num_stages=1
        )

        # 3) Triton: G contraction G[b, n, i, j, h] = sum_d C[b, n, i, h, d] * B[b, n, j, h, d]
        G = torch.empty((B_size, N_size, S_size, S_size, H_size), dtype=torch.float32, device=hidden_states.device)

        C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d = C.stride()
        B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d = B.stride()
        G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h = G.stride()


def run(*args):
    return ModelNew()(*args)
