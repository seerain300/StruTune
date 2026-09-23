import torch
import triton
import triton.language as tl


# Kernel 1: Compute L = exp(cumsum(masked A)) after expanding A_cumsum to [S, S] and applying tril(diagonal=-1).
# Input: A_cumsum [B, H, N, S], Output: L [B, H, N, S, S] (float32)
@triton.jit
def a_tril_cumsum_exp_kernel(
    A_ptr, L_ptr,
    B_size, H_size, N_size, S_size,
    A_stride_b, A_stride_h, A_stride_n, A_stride_s,
    L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
):
    # Each program handles one (b, h, n)
    pid_b = tl.program_id(0)  # b
    pid_h = tl.program_id(1)  # h
    pid_n = tl.program_id(2)  # n

    # We will fill the SxS matrix L[b,h,n,i,j] = exp(sum_{k=0..i} A[b,h,n,k]) if j<=i else 0
    # Note: tril(diagonal=-1) means only j <= i should have non-zero values.
    # We create a 2D tile over (i, j) inside this program, as S is passed as a constexpr.
    # We'll use static loops for i and j in [0..S-1].
    # prefix accumulates sum over k in [0..i]; for j > i, store 0.
    for i in range(S_size):
        prefix = 0.0
        for k in range(i + 1):  # sum over k from 0 to i
            a_val = tl.load(A_ptr + pid_b * A_stride_b + pid_h * A_stride_h + pid_n * A_stride_n + k * A_stride_s)
            prefix += a_val
        # For each j <= i: L[i,j] = exp(prefix)
        # For j > i: L[i,j] = 0
        for j in range(S_size):
            if j <= i:
                l_val = tl.exp(prefix)
            else:
                l_val = 0.0
            # Store to L[b,h,n,i,j] using its stride
            tl.store(
                L_ptr + pid_b * L_stride_b + pid_h * L_stride_h + pid_n * L_stride_n + i * L_stride_s1 + j * L_stride_s2,
                l_val
            )


# Kernel 2: Compute G[b,n,i,j,h] = sum_d C[b,n,i,h,d] * B[b,n,j,h,d]
@triton.jit
def g_contract_kernel(
    B_ptr, C_ptr, G_ptr,
    B_size, N_size, S, H, D,
    B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
    C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)  # b
    pid_n = tl.program_id(1)  # n
    pid_i = tl.program_id(2)  # i in [0..S-1]
    pid_j = tl.program_id(3)  # j in [0..S-1]
    pid_h = tl.program_id(4)  # h in [0..H-1]

    acc = 0.0  # sum over D
    # Tile over D
    for d_start in range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        mask = d_offsets < D
        # Load B[b,n,j,h,d] and C[b,n,i,h,d]
        B_vec = tl.load(
            B_ptr + pid_b * B_stride_b + pid_n * B_stride_n + pid_j * B_stride_s + pid_h * B_stride_h + d_offsets * B_stride_d,
            mask=mask, other=0.0
        )
        C_vec = tl.load(
            C_ptr + pid_b * C_stride_b + pid_n * C_stride_n + pid_i * C_stride_s + pid_h * C_stride_h + d_offsets * C_stride_d,
            mask=mask, other=0.0
        )
        prod = B_vec * C_vec
        # Reduce within the tile to a scalar
        # Simple loop reduction:
        for k in range(BLOCK_D):
            acc += prod[k]
    # Store G[b,n,i,j,h] = acc
    tl.store(
        G_ptr + pid_b * G_stride_b + pid_n * G_stride_n + pid_i * G_stride_s1 + pid_j * G_stride_s2 + pid_h * G_stride_h,
        acc
    )


# Kernel 3: Elementwise M = G * L
@triton.jit
def m_mul_kernel(
    G_ptr, L_ptr, M_ptr,
    B_size, N_size, S, H,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
):
    pid_b = tl.program_id(0)  # b
    pid_n = tl.program_id(1)  # n
    pid_i = tl.program_id(2)  # i
    pid_j = tl.program_id(3)  # j
    pid_h = tl.program_id(4)  # h

    g_val = tl.load(G_ptr + pid_b * G_stride_b + pid_n * G_stride_n + pid_i * G_stride_s1 + pid_j * G_stride_s2 + pid_h * G_stride_h)
    l_val = tl.load(L_ptr + pid_b * L_stride_b + pid_h * L_stride_h + pid_n * L_stride_n + pid_i * L_stride_s1 + pid_j * L_stride_s2)
    m_val = g_val * l_val
    tl.store(M_ptr + pid_b * M_stride_b + pid_n * M_stride_n + pid_i * M_stride_s1 + pid_j * M_stride_s2 + pid_h * M_stride_h, m_val)


# Kernel 4: Y_diag[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden_states[b,n,j,h] over D
@triton.jit
def y_diag_reduce_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    B_size, N_size, S, H, D,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
    hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
    Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)  # b
    pid_n = tl.program_id(1)  # n
    pid_i = tl.program_id(2)  # i
    pid_h = tl.program_id(3)  # h

    acc = 0.0
    # Loop over j dimension in tiles
    for j_start in range(0, S, BLOCK_D):
        j_offsets = j_start + tl.arange(0, BLOCK_D)
        mask_j = j_offsets < S
        # Compute dot-product over D for each j in the tile
        for jj in range(BLOCK_D):
            j = j_start + jj
            if j < S:
                # Load M[b,n,i,j,h] as a scalar
                m_val = tl.load(M_ptr + pid_b * M_stride_b + pid_n * M_stride_n + pid_i * M_stride_s1 + j * M_stride_s2 + pid_h * M_stride_h)
                # Load hidden[b,n,j,h,:] vector over D
                h_vec = tl.load(
                    hidden_ptr + pid_b * hidden_stride_b + pid_n * hidden_stride_n + j * hidden_stride_s + pid_h * hidden_stride_h + tl.arange(0, BLOCK_D) * hidden_stride_d,
                    mask=tl.arange(0, BLOCK_D) < D, other=0.0
                )
                # For this single j, accumulate sum over D: m_val * sum_d hidden[b,n,j,h,d]
                # But hidden is a vector over BLOCK_D; we must sum over valid D entries.
                # We can't sum a vector directly; instead, accumulate scalar contributions by looping small BLOCK_D (here D is usually small).
                # For simplicity, loop over D scalarly:
                # Note: This kernel assumes D is small; in this benchmark, D=32.
                for k in range(D):
                    h_elem = tl.load(hidden_ptr + pid_b * hidden_stride_b + pid_n * hidden_stride_n + j * hidden_stride_s + pid_h * hidden_stride_h + k * hidden_stride_d)
                    acc += m_val * h_elem
    tl.store(Y_ptr + pid_b * Y_stride_b + pid_n * Y_stride_n + pid_i * Y_stride_s1 + pid_h * Y_stride_h, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; all computation in kernels

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Ensure on CUDA and contiguous
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA device"
        hidden = hidden_states.contiguous()
        A = A_cumsum.contiguous()
        Bx = B.contiguous()
        Cx = C.contiguous()

        # Shapes
        B_size, N_size, S, H, D = hidden.shape
        assert A.shape == (B_size, H, N_size, S), "A_cumsum must have shape [B, H, N, S]"
        # The original code expands B/C from G to H via repeat_interleave(NUM_HEADS // N_GROUPS), which is 4.
        # Here we assume B/C are already expanded to H; if not, they should be passed expanded.
        assert Bx.shape[3] == H and Cx.shape[3] == H, "B/C must be expanded to num_heads (H)"

        device = hidden.device

        # Output Y_diag: [B, N, S, H] in float32 (then cast to bfloat16 as original returns bfloat16)
        # 1) Compute L in Triton: L[b,h,n,i,j] = exp(cumsum(masked A_expanded)) with tril(diagonal=-1)
        L = torch.empty((B_size, H, N_size, S, S), dtype=torch.float32, device=device)

        A_stride_b, A_stride_h, A_stride_n, A_stride_s = A.stride()
        L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2 = L.stride()

        # Launch a_tril_cumsum_exp_kernel over (B,H,N)
        grid_a = (B_size, H_size, N_size)
        a_tril_cumsum_exp_kernel[grid_a](
            A, L,
            B_size, H_size, N_size, S,
            A_stride_b, A_stride_h, A_stride_n, A_stride_s,
            L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
            num_warps=1, num_stages=1
        )

        # 2) Compute G[b,n,i,j,h] via Triton
        G = torch.empty((B_size, N_size, S, S, H), dtype=torch.float32, device=device)
        B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d = Bx.stride()
        C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d = Cx.stride()
        G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h = G.stride()

        grid_g = (B_size, N_size, S, S, H)
        g_contract_kernel[grid_g](
            Bx, Cx, G,
            B_size, N_size, S, H, D,
            B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
            C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            BLOCK_D=32,
            num_warps=1, num_stages=1
        )

        # 3) Elementwise M = G * L via Triton
        M = torch.empty((B_size, N_size, S, S, H), dtype=torch.float32, device=device)
        L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2 = L.stride()
        M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h = M.stride()

        grid_m = (B_size, N_size, S, S, H)
        m_mul_kernel[grid_m](
            G, L, M,
            B_size, N_size, S, H,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
            M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
            num_warps=1, num_stages=1
        )

        # 4) Compute Y_diag[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden[b,n,j,h] via Triton
        Y = torch.empty((B_size, N_size, S, H), dtype=torch.float32, device=device)

        M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h = M.stride()
        hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d = hidden.stride()
        Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h = Y.stride()

        grid_y = (B_size, N_size, S, H)
        y_diag_reduce_kernel[grid_y](
            M, hidden, Y,
            B_size, N_size, S, H, D,
            M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
            hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
            Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h,
            BLOCK_D=32,
            num_warps=1, num_stages=1
        )

        # Return in bfloat16 to match original behavior
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
