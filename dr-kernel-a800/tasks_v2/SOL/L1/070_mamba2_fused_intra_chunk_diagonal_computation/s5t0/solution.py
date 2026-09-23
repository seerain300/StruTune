import torch
import triton
import triton.language as tl

# Constants from the original code
CHUNK_SIZE = 128
NUM_HEADS = 32
N_GROUPS = 8

@triton.jit
def l_causal_kernel(
    A_ptr,          # *const float, shape [B, C, L, H]
    L_ptr,          # *float,       shape [B, C, H, L, L]
    B: tl.constexpr,  # batch size
    C: tl.constexpr,  # num_chunks
    H: tl.constexpr,  # num_heads
    L_size: tl.constexpr,  # chunk size (CHUNK_SIZE)
):
    """
    For each (b, c, h), build L[b, c, h, :, :] with lower-triangular causal mask (diagonal = -1).
    L[i, j] = exp(sum_{k=0..i-1} A[b, c, k, h]) for i > j (since i >= j is masked), else 0.
    We implement the masking by setting upper triangle to 0, then exp() on the matrix.
    """
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # Ensure we don't go out of bounds on program_id
    if (b >= B) or (c >= C) or (h >= H):
        return

    # Create 2D offsets for rows (i) and cols (j)
    rows = tl.arange(0, L_size)
    cols = tl.arange(0, L_size)

    # Build mask for lower triangle (exclude diagonal), j < i
    lower_mask = cols[None, :] < rows[:, None]
    # Pointer to start of this (b, c, h) block in A
    base = (b * C * H + c * H + h) * L_size * L_size  # since A is [B, C, L, H] contiguous

    # Initialize L with zeros
    L_mat = tl.zeros((L_size, L_size), dtype=tl.float32)

    # For each row i, compute sum over k=0..i-1 of A[b, c, k, h]
    # and fill L_mat[j, i] = exp(sum) for j < i
    # Note: Triton loop over static bounds is fine here.
    for i in range(L_size):
        # sum up A[b, c, k, h] for k in [0, i-1]
        s = 0.0
        for k in range(i):
            a_val = tl.load(A_ptr + base + k * L_size + i)  # A is [B, C, L, H], contiguous -> index = b*C*L*H + c*H*L + k*L + h
            s += a_val
        # Fill lower triangle positions (j < i) with exp(s)
        # We do it column-wise for j=0..L_size-1
        for j in range(L_size):
            if lower_mask[i, j]:
                L_mat[j, i] = tl.exp(s)
            else:
                L_mat[j, i] = 0.0

    # Store L_mat to L[b, c, h, :, :]
    L_base = (b * C * H + c * H + h) * L_size * L_size
    # L is [B, C, H, L, L] contiguous => offset = (b*C*H + c*H + h) * L*L + i*L + j
    for i in range(L_size):
        for j in range(L_size):
            tl.store(L_ptr + L_base + i * L_size + j, L_mat[j, i])

@triton.jit
def g_outer_kernel(
    B_ptr,  # *const float, shape [B, C, L, H*GROUP_EXPAND]
    C_ptr,  # *const float, shape [B, C, L, H*GROUP_EXPAND]
    G_ptr,  # *float,       shape [B, C, L, L, H]
    B: tl.constexpr,  # batch size
    C: tl.constexpr,  # num_chunks
    H: tl.constexpr,  # num_heads
    L: tl.constexpr,  # chunk size (CHUNK_SIZE)
    GROUP_EXPAND: tl.constexpr,  # num_heads // (N_GROUPS)
    S: tl.constexpr,  # state_size (B/C last dim)
    BLOCK_S: tl.constexpr  # reduction block for state
):
    """
    For each (i, j, h), compute G[i, j, h] = sum over s of C[i, s, h] * B[j, s, h].
    Launch grid over (i, j, h). We collapse (i, j) into a single program_id for simplicity.
    """
    # program_id(0) over (i, j) flattened; program_id(1) over h
    pid0 = tl.program_id(0)
    h = tl.program_id(1)

    # Map pid0 to (i, j)
    # We'll use a single grid dimension for (i, j). Since i and j each up to L, pid0 in [0, L*L)
    # But Triton grid must be known. We'll set grid=(B*C*L*L, H) and inside compute i,j:
    # However, Triton requires static loops; better to use 3D grid: (i, j, h).
    # Since Triton kernel launch grid is static, we reconstruct i,j from pid0 via modulo:
    # We need to pass C into kernel? In our launch we can set grid=(B*C*L*L, H), then:
    # i = pid0 // L, j = pid0 % L
    i = pid0 // L
    j = pid0 % L

    # Bounds check
    if (i >= L) or (j >= L) or (h >= H):
        return

    # We need to iterate over s in [0, S) and accumulate C[i, s, h] * B[j, s, h]
    acc = 0.0
    for s0 in range(0, S, BLOCK_S):
        s_idx = s0 + tl.arange(0, BLOCK_S)
        mask = s_idx < S
        # Load B[j, s, h] vector and C[i, s, h] vector
        # For B_ptr: index = b*C*L*(H*GROUP_EXPAND) + c*L*(H*GROUP_EXPAND) + j*(H*GROUP_EXPAND) + h*(GROUP_EXPAND) + s
        # We don't have batch/num_chunks here, so we rely on caller passing B/C that are already expanded to H.
        # We assume B_ptr and C_ptr are [B, C, L, H] already expanded.
        # We need to know b and c; Triton kernels don't get batch/num_chunks scalars unless passed.
        # Therefore, we will launch with grid over (B*C*L*L, H) and use program_id(2) to pass b,c.
        # But Triton kernel signature doesn't support program_id(2) here. So we instead write kernel with 2D grid and
        # pass b,c as program_id(0), (i,j) as program_id(1), h as program_id(2). Simplify: use 3D grid.
        # To keep it simple, we redefine kernel signature to accept B, C, H as constexpr and also pass b,c via program_id(0).

        # We'll re-implement kernel with proper 3D grid below. For now, exit.

@triton.jit
def g_outer_kernel_3d(
    B_ptr,  # *const float, shape [B, C, L, H]
    C_ptr,  # *const float, shape [B, C, L, H]
    G_ptr,  # *float,       shape [B, C, L, L, H]
    B: tl.constexpr,  # batch size
    C: tl.constexpr,  # num_chunks
    H: tl.constexpr,  # num_heads
    L: tl.constexpr,  # chunk size (CHUNK_SIZE)
    S: tl.constexpr,  # state_size (B/C last dim)
    BLOCK_S: tl.constexpr  # reduction block for state
):
    """
    3D grid: (b, c, i). Inside kernel, loop j and h to compute G[i, j, h].
    However, Triton supports only 3 grid dims. To compute for all (i, j, h), we use a single 3D grid
    over (b, c, i) and inside loop j and h. This is fine for small L (128).
    """
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)

    if (b >= B) or (c >= C) or (i >= L):
        return

    # Initialize acc over h
    for h in range(H):
        acc = 0.0
        # Loop over s in chunks
        for s0 in range(0, S, BLOCK_S):
            s_idx = s0 + tl.arange(0, BLOCK_S)
            mask = s_idx < S
            # Load B[j, s, h] for all j in parallel? Not feasible since j varies per iteration.
            # Instead, loop j and accumulate scalar:
            for j in range(L):
                # Compute sum over s of C[i, s, h] * B[j, s, h]
                # For each j, we need to sum C[i, s, h] * B[j, s, h] over s.
                # We'll do scalar loads for s to avoid too many vector accumulators.
                g_val = 0.0
                for s in range(0, S):
                    c_val = tl.load(C_ptr + b * C * L * H + c * L * H + i * H + h * S + s)
                    b_val = tl.load(B_ptr + b * C * L * H + c * L * H + j * H + h * S + s)
                    g_val += c_val * b_val
                # Store G[b, c, i, j, h] = g_val
                tl.store(G_ptr + b * C * L * L * H + c * L * L * H + i * L * H + j * H + h, g_val)

@triton.jit
def y_diag_reduce_kernel(
    M_ptr,            # *const float, shape [B, C, L, L, H]
    HS_ptr,           # *const float, shape [B, C, L, H, D]
    Y_ptr,            # *float,       shape [B, C, L, H, D]
    B: tl.constexpr,  # batch size
    C: tl.constexpr,  # num_chunks
    L: tl.constexpr,  # chunk size (CHUNK_SIZE)
    H: tl.constexpr,  # num_heads
    D: tl.constexpr,  # head_dim
    BLOCK_D: tl.constexpr  # block for head_dim reduction
):
    """
    For each (b, c, i, h), reduce over j: Y[b, c, i, h, :] = sum_j M[b, c, i, j, h] * HS[b, c, j, h, :].
    Launch one program per (b, c, i, h). Vectorize along head_dim D.
    """
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    if (b >= B) or (c >= C) or (i >= L) or (h >= H):
        return

    # Accumulator over head_dim
    # We'll accumulate into a vector of size D
    acc = tl.zeros((D,), dtype=tl.float32)

    # Loop over j from 0 to L-1
    for j in range(L):
        # Load M[b, c, i, j, h] as scalar
        m_val = tl.load(M_ptr + b * C * L * L * H + c * L * L * H + i * L * H + j * H + h)
        # Load HS[b, c, j, h, :] as vector
        d_idx = tl.arange(0, BLOCK_D)
        mask_d = d_idx < D
        hs_vec = tl.load(HS_ptr + b * C * L * H * D + c * L * H * D + j * H * D + h * D + d_idx, mask=mask_d, other=0.0)
        # Multiply and accumulate
        acc += m_val * hs_vec

    # Store acc to Y[b, c, i, h, :]
    y_base = b * C * L * H * D + c * L * H * D + i * H * D + h * D
    tl.store(Y_ptr + y_base + tl.arange(0, D), acc, mask=mask_d)

# ModelNew: Triton-optimized version of the original run function
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; we will rely on input tensors

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Compute intra-chunk diagonal output Y_diag for Mamba2 SSD using Triton kernels.
        Inputs:
          - hidden_states: [B, C, L, H, D]
          - A_cumsum:      [B, C, L, H] (we will expand to [B, C, L, H, L] in kernel)
          - B:             [B, C, L, G, S] where G=N_GROUPS, S=state_size
          - C:             [B, C, L, G, S]
        Returns:
          - Y_diag: [B, C, L, H, D] in bfloat16
        """
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Triton requires CUDA tensors"
        # Ensure contiguity
        hidden_states = hidden_states.contiguous()
        A_cumsum = A_cumsum.contiguous()
        B = B.contiguous()
        C = C.contiguous()

        B_size, C_size, L, H, D = hidden_states.shape
        # We use CHUNK_SIZE = 128, NUM_HEADS = 32, N_GROUPS = 8. Note: The original code uses NUM_HEADS=32, N_GROUPS=8,
        # but in the original run() function, hidden_states.shape uses NUM_HEADS=32. We keep that.
        GROUP_EXPAND = H // 8  # since original NUM_HEADS=32, N_GROUPS=8 -> 4
        # Expand B and C from n_groups to num_heads (host-side metadata ops)
        B_expanded = B.repeat_interleave(GROUP_EXPAND, dim=3)  # [B, C, L, H, S]
        C_expanded = C.repeat_interleave(GROUP_EXPAND, dim=3)  # [B, C, L, H, S]

        # Compute L (causal mask) in Triton: shape [B, C, H, L, L] in float32
        L_mat = torch.empty((B_size, C_size, H, L, L), device=hidden_states.device, dtype=torch.float32)
        # Launch l_causal_kernel: grid = (B, C, H)
        # Note: Triton uses meta-parameters. We set num_warps=4 for 128x128 work
        l_causal_kernel[(B_size, C_size, H)](A_cumsum, L_mat, B_size, C_size, H, L, num_warps=4)

        # Compute G in Triton: G[b, c, i, j, h] = sum_s C_expanded[b, c, i, h, s] * B_expanded[b, c, j, h, s]
        G = torch.empty((B_size, C_size, L, L, H), device=hidden_states.device, dtype=torch.float32)
        S = C_expanded.shape[-1]
        # Choose BLOCK_S for reduction
        BLOCK_S = 128
        # Launch g_outer_kernel_3d: grid = (B, C, L)
        g_outer_kernel_3d[(B_size, C_size, L)](B_expanded, C_expanded, G, B_size, C_size, H, L, S, BLOCK_S, num_warps=4)

        # Compute M = G * L_permuted (L originally is [B, C, H, L, L], we need [B, C, L, L, H])
        # We can use torch for permutation here (metadata op), since it's a view:
        L_perm = L_mat.permute(0, 1, 2, 3, 4)  # [B, C, H, L, L] -> [B, C, L, L, H] after permute is not correct; fix:
        # Actually we want to get L in [B, C, L, L, H] which is L_mat already. So no need to permute.
        M = G * L_mat  # broadcast multiply over H dim

        # Compute Y_diag in Triton: Y_diag[b, c, i, h, d] = sum_j M[b, c, i, j, h] * HS[b, c, j, h, d]
        # We need HS to be [B, C, L, H, D]; hidden_states already has that shape.
        Y = torch.empty((B_size, C_size, L, H, D), device=hidden_states.device, dtype=torch.float32)
        BLOCK_D = 128  # we'll support D up to 128; if smaller, masked stores work
        y_diag_reduce_kernel[(B_size, C_size, L, H)](M, hidden_states, Y, B_size, C_size, L, H, D, BLOCK_D, num_warps=4)

        # Return in bfloat16, matching original
        return Y.to(torch.bfloat16)

# Optional: if you want to test, you can create a small example and run:
# model = ModelNew().cuda()
# # Prepare dummy inputs (ensure chunk_size == 128)
# B, C, L, H, D = 2, 2, 128, 32, 64
# hidden_states = torch.randn(B, C, L, H, D, device='cuda', dtype=torch.float32)
# A_cumsum = torch.randn(B, C, L, H, device='cuda', dtype=torch.float32)  # shape should be [B, C, L, H]; original uses [B, C, L, H]
# B_t = torch.randn(B, C, L, 8, 32, device='cuda', dtype=torch.float32)
# C_t = torch.randn(B, C, L, 8, 32, device='cuda', dtype=torch.float32)
# out = model(hidden_states, A_cumsum, B_t, C_t)
# print(out.shape)  # should be [B, C, L, H, D] in bfloat16


def run(*args):
    return ModelNew()(*args)
