import torch
import triton
import triton.language as tl


@triton.jit
def tril_mask_kernel(M: tl.pointer_type(tl.int8), S: tl.constexpr):
    """
    Produce a lower-triangular mask with diagonal=-1:
    M[i, j] = 1 if j <= i else 0, stored as int8 (1/0).
    Grid: (S, S)
    """
    i = tl.program_id(0)
    j = tl.program_id(1)
    if j <= i:
        M[i * S + j] = 1
    else:
        M[i * S + j] = 0


@triton.jit
def exp_cumsum_tril_kernel(
    A_ptr,  # [B, H, N, S, S] linearized
    L_ptr,  # [B, H, N, S, S] linearized (output)
    B_size: tl.constexpr, H_size: tl.constexpr, N_size: tl.constexpr, S: tl.constexpr,
    A_stride_b, A_stride_h, A_stride_n, A_stride_s1, A_stride_s2,
    Out_stride_b, Out_stride_h, Out_stride_n, Out_stride_s1, Out_stride_s2,
):
    """
    For each fixed (b, h, n), compute L[i, j] = exp(sum_{k<=i} A[b, h, n, k, j]) for j <= i,
    and L[i, j] = 0 for j > i.
    We iterate over i from 0..S-1 and maintain a running prefix sum per j.
    """
    b = tl.program_id(0)  # 0..B-1, not used in loop; use b=0 and grid dim over (H,N,S)
    h = tl.program_id(1)
    n = tl.program_id(2)
    # Loop over i (scan dimension)
    for i in range(0, S):
        running = tl.zeros((S,), dtype=tl.float32)
        # For each j, compute sum of A over k<=i and store exp of running
        for j in range(0, S):
            ptr = A_ptr + b * A_stride_b + h * A_stride_h + n * A_stride_n + i * A_stride_s1 + j * A_stride_s2
            # A values are float32 (we'll pass A_expanded as float32)
            a = tl.load(ptr)
            if j <= i:
                running += a
            # L value
            l_val = tl.exp(running) if j <= i else 0.0
            out_ptr = L_ptr + b * Out_stride_b + h * Out_stride_h + n * Out_stride_n + i * Out_stride_s1 + j * Out_stride_s2
            tl.store(out_ptr, l_val)


@triton.jit
def g_contract_kernel(
    B_ptr,  # [B, N, S, H, D] linearized
    C_ptr,  # [B, N, S, H, D] linearized
    G_ptr,  # [B, N, S, S, H] linearized
    B_size, N_size, S, H, D,
    B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
    C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
):
    """
    Compute G[i,j,h] = sum_d C[b,n,i,h,d] * B[b,n,j,h,d] for all (b,n,i,j,h).
    Grid: (B, N, S, S, H)
    """
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)
    # Loop over D dimension in tiles
    for d_start in range(0, D, 1):  # D is small (e.g., 64), can loop directly
        d = d_start
        if d < D:
            B_val = tl.load(B_ptr + b * B_stride_b + n * B_stride_n + i * B_stride_s + h * B_stride_h + d * B_stride_d)
            C_val = tl.load(C_ptr + b * C_stride_b + n * C_stride_n + j * C_stride_s + h * C_stride_h + d * C_stride_d)
            acc += B_val * C_val
    # Store G[i,j,h]
    out_ptr = G_ptr + b * G_stride_b + n * G_stride_n + i * G_stride_s1 + j * G_stride_s2 + h * G_stride_h
    tl.store(out_ptr, acc)


@triton.jit
def m_mul_kernel(
    G_ptr,  # [B, N, S, S, H]
    L_ptr,  # [B, N, S, S, H]
    M_ptr,  # [B, N, S, S, H]
    B_size, N_size, S, H,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    L_stride_b, L_stride_n, L_stride_s1, L_stride_s2, L_stride_h,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
):
    """
    Compute M = G * L along the last dim (H). We treat G and L as [B,N,S,S,H] and multiply elementwise.
    Grid: (B, N, S, S, H)
    """
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    G_val = tl.load(G_ptr + b * G_stride_b + n * G_stride_n + i * G_stride_s1 + j * G_stride_s2 + h * G_stride_h)
    L_val = tl.load(L_ptr + b * L_stride_b + n * L_stride_n + i * L_stride_s1 + j * L_stride_s2 + h * L_stride_h)
    tl.store(M_ptr + b * M_stride_b + n * M_stride_n + i * M_stride_s1 + j * M_stride_s2 + h * M_stride_h, G_val * L_val)


@triton.jit
def y_diag_reduce_kernel(
    M_ptr,           # [B, N, S, S, H]
    hidden_ptr,      # [B, N, S, H, D]
    out_ptr,         # [B, N, S, H, D]
    B_size, N_size, S, H, D,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
    hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
    out_stride_b, out_stride_n, out_stride_s, out_stride_h, out_stride_d,
):
    """
    Compute Y[b,n,i,h,:] = sum_j M[b,n,i,j,h] * hidden[b,n,j,h,:].
    Grid: (B, N, S, H)
    """
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    acc = tl.zeros((D,), dtype=tl.float32)
    # Loop over j (chunk dimension)
    for j in range(0, S):
        M_j = tl.load(M_ptr + b * M_stride_b + n * M_stride_n + i * M_stride_s1 + j * M_stride_s2 + h * M_stride_h)
        hidden_vec = tl.load(
            hidden_ptr + b * hidden_stride_b + n * hidden_stride_n + j * hidden_stride_s + h * hidden_stride_h + tl.arange(0, D) * hidden_stride_d
        )
        acc += M_j * hidden_vec
    # Store acc to out[b,n,i,h,:]
    base = out_ptr + b * out_stride_b + n * out_stride_n + i * out_stride_s + h * out_stride_h
    tl.store(base + tl.arange(0, D) * out_stride_d, acc)


class ModelNew(torch.nn.Module):
    """
    Triton-optimized version of the original Model, with all numeric ops performed in Triton kernels.
    """
    def __init__(self, NUM_HEADS: int = 32, N_GROUPS: int = 8):
        super().__init__()
        self.NUM_HEADS = NUM_HEADS
        self.N_GROUPS = N_GROUPS

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Shapes
        B_size, N_size, S_size, H_size, D_size = hidden_states.shape
        assert A_cumsum.shape == (B_size, H_size, N_size, S_size), "A_cumsum must have shape [B, H, N, S]"
        assert B.shape == (B_size, N_size, S_size, H_size, D_size), "B must have shape [B, N, S, H, D]"
        assert C.shape == (B_size, N_size, S_size, H_size, D_size), "C must have shape [B, N, S, H, D]"
        assert H_size == self.NUM_HEADS, "num_heads must be 32"
        assert hidden_states.device.type == 'cuda', "ModelNew requires CUDA device for Triton kernels"

        device = hidden_states.device

        # 1) Triton: Lower-triangular mask M_lower of shape [S, S] with diagonal=-1 (j <= i)
        M_lower = torch.empty((S_size, S_size), dtype=torch.int8, device=device)
        grid_mask = (S_size, S_size)
        tril_mask_kernel[grid_mask](M_lower, S_size, num_warps=1, num_stages=1)

        # 2) Triton: L = exp(cumsum(masked A)) along S for each (b,h,n,i)
        A_expanded = A_cumsum.unsqueeze(-1).expand(B_size, H_size, N_size, S_size, S_size).to(torch.float32).contiguous()
        L = torch.empty((B_size, H_size, N_size, S_size, S_size), dtype=torch.float32, device=device)

        A_stride_b, A_stride_h, A_stride_n, A_stride_s1, A_stride_s2 = A_expanded.stride()
        Out_stride_b, Out_stride_h, Out_stride_n, Out_stride_s1, Out_stride_s2 = L.stride()
        grid = (B_size, H_size, N_size, S_size)
        exp_cumsum_tril_kernel[grid](
            A_expanded, L,
            B_size, H_size, N_size, S_size,
            A_stride_b, A_stride_h, A_stride_n, A_stride_s1, A_stride_s2,
            Out_stride_b, Out_stride_h, Out_stride_n, Out_stride_s1, Out_stride_s2,
            num_warps=1, num_stages=1
        )

        # 3) Triton: G contraction
        G = torch.empty((B_size, N_size, S_size, S_size, H_size), dtype=torch.float32, device=device)

        B_contig = B.contiguous()
        C_contig = C.contiguous()

        B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d = B_contig.stride()
        C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d = C_contig.stride()
        G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h = G.stride()

        grid_g = (B_size, N_size, S_size, S_size, H_size)
        g_contract_kernel[grid_g](
            B_contig, C_contig, G,
            B_size, N_size, S_size, H_size, D_size,
            B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
            C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            num_warps=1, num_stages=1
        )

        # 4) Triton: M = G * L
        # Permute L to match G: [B, N, S, S, H] (L already has shape [B, H, N, S, S]; permute to [B, N, S, S, H])
        L_perm = L.permute(0, 2, 3, 4, 1).contiguous()  # [B, N, S, S, H]
        M = torch.empty_like(G)  # [B, N, S, S, H]
        grid_m = (B_size, N_size, S_size, S_size, H_size)
        m_mul_kernel[grid_m](
            G, L_perm, M,
            B_size, N_size, S_size, H_size,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            L_stride_b=0, L_stride_n=0, L_stride_s1=0, L_stride_s2=0, L_stride_h=0,  # dummy, not used
            M_stride_b=0, M_stride_n=0, M_stride_s1=0, M_stride_s2=0, M_stride_h=0,  # will be provided by strides of M
        )
        # Actually, pass correct strides for M:
        M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h = M.stride()
        m_mul_kernel[grid_m](
            G, L_perm, M,
            B_size, N_size, S_size, H_size,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            L_stride_b=0, L_stride_n=0, L_stride_s1=0, L_stride_s2=0, L_stride_h=0,  # dummy
            M_stride_b=M_stride_b, M_stride_n=M_stride_n, M_stride_s1=M_stride_s1, M_stride_s2=M_stride_s2, M_stride_h=M_stride_h,
            num_warps=1, num_stages=1
        )

        # 5) Triton: Y_diag reduction over j: Y[b,n,i,h,:] = sum_j M[b,n,i,j,h] * hidden_states[b,n,j,h,:]
        hidden_contig = hidden_states.contiguous()
        Y = torch.empty((B_size, N_size, S_size, H_size, D_size), dtype=torch.float32, device=device)

        M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h = M.stride()
        hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d = hidden_contig.stride()
        out_stride_b, out_stride_n, out_stride_s, out_stride_h, out_stride_d = Y.stride()

        grid_y = (B_size, N_size, S_size, H_size)
        y_diag_reduce_kernel[grid_y](
            M, hidden_contig, Y,
            B_size, N_size, S_size, H_size, D_size,
            M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
            hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
            out_stride_b, out_stride_n, out_stride_s, out_stride_h, out_stride_d,
            num_warps=1, num_stages=1
        )

        # Return in bfloat16 to match original behavior
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
