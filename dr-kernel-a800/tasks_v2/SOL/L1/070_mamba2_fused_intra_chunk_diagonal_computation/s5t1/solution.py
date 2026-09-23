import torch
import triton
import triton.language as tl

# Kernel 1: Build L matrix per (b, c, h) with lower-triangular causal mask after cumsum along L (128)
@triton.jit
def l_causal_kernel(
    A_ptr,          # *float32, shape [B, C, L, H]
    L_ptr,          # *float32, shape [B, C, H, L, L]
    B_size, C_size, H, L,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # Indices for rows and cols
    rows = tl.arange(0, L)[:, None]  # shape [L, 1]
    cols = tl.arange(0, L)[None, :]  # shape [1, L]

    # Compute address of A_cumsum[b, c, :, h]
    # A strides: (stride_b, stride_c, stride_l, stride_h)
    # We need to get base pointer for this (b, c, h), then load along L
    # Strides are not passed; assume A is contiguous in L,H and in B,C as batched.
    # We can infer from A_ptr: A is [B, C, L, H], so we access A[b, c, :, h].
    # Create a base pointer for A[b, c, :, h]: increment by b*stride_b + c*stride_c + h*stride_h, then iterate l.
    # However, Triton doesn't let us access strides directly; we instead rely on contiguous layout by passing flattened pointers.
    # To keep it simple, we pass A as contiguous [B, C, L, H] so that for fixed (b,c,h), A[b,c,:,h] is a contiguous vector of length L.
    # We'll assume A is contiguous in memory: offset = b*C_size*L*H + c*L*H + l*H + h
    # But better: we pass A as [B, C, L, H] contiguous, then for fixed (b,c,h), A[b,c,:,h] is contiguous.
    # We'll launch with A already contiguous and compute base offset: b*C*L*H + c*L*H + h*L, then iterate l.

    # Compute base offset for A[b, c, :, h] contiguous
    base_AC = b * C_size * L * H + c * L * H + h * L

    # Load A values along rows
    # For each row i, we need to load A[b, c, i, h]
    # We will compute a vector of offsets for rows: base_AC + rows * H + h * 0 => base_AC + rows * H
    # But rows is index into L; offset for A[b, c, rows, h] is base_AC + rows * H + h * (L if false, but H stride for last dim is 1? In contiguous [B,C,L,H], stride for H is L; last dim is H with size H, so stride for H is L? Actually for [B,C,L,H], strides are (C*L*H, L*H, H, 1).
    # To be precise, for contiguous [B,C,L,H], stride for L is H, stride for H is 1. So address for A[b, c, l, h] is base_AC + l*H + h.
    # Therefore, load A_row[i] = tl.load(A_ptr + base_AC + rows * H + h).
    # However, base_AC above only handles c. We need to account for C stride: stride for C is L*H.
    # But since we launch with fixed (b,c), we only need to add c*L*H. We already did c*L*H above.
    # So A[b, c, rows, h] address: A_ptr + base_AC + rows * H + h
    # Note: h is scalar, so we can add h directly. For contiguous [B,C,L,H], last dim H has stride 1.
    # Thus: A_row = tl.load(A_ptr + base_AC + rows * H + h)
    # Compute cumsum along rows (i dimension): we need per-row cumsum. Triton allows loops.
    # We will compute cumsum[i] for each i by summing A[b, c, k, h] for k < i.

    # Initialize cumsum vector for rows: length L
    cumsum_rows = tl.zeros((L,), dtype=tl.float32)
    # We can't directly use rows as indices for load in vectorized way; Triton supports while loops.
    # We need to compute cumsum[i] for i in 0..L-1
    i = 0
    while i < L:
        # Load A[b, c, i, h]
        # Address: A_ptr + base_AC + i*H + h
        # But base_AC already includes c*L*H + h; so address is A_ptr + base_AC + i*H
        A_i = tl.load(A_ptr + base_AC + i * H)
        cumsum_rows[i] = A_i
        # For k < i, add A[k]
        k = 0
        while k < i:
            A_k = tl.load(A_ptr + base_AC + k * H)
            cumsum_rows[i] += A_k
            k += 1
        i += 1

    # Now build L matrix: for each (i, j), if j < i, L[i, j] = exp(cumsum_rows[i] - cumsum_rows[j]); else 0
    # We will fill L_ptr[b, c, h, i, j]
    # Base offset for L[b, c, h] is: b*(C_size*H*L*L) + c*(H*L*L) + h*(L*L)
    # But Triton kernel pointer arithmetic uses linear indexing. Simpler: we assume L is contiguous with shape [B, C, H, L, L]
    # For fixed (b,c,h), L is a contiguous 2D block of size L*L. The stride for H is L*L, for L (row) is L, for L (col) is 1.
    # So address for L[b, c, h, i, j] = ((b*C_size + c)*H + h) * (L*L) + i*L + j
    # We can compute this linear index.
    # We need to loop i and j.
    i_idx = 0
    while i_idx < L:
        j_idx = 0
        while j_idx < L:
            # Compute linear index for L[b, c, h, i_idx, j_idx]
            L_linear_index = ((b * C_size + c) * H + h) * (L * L) + i_idx * L + j_idx
            # Set value if j_idx < i_idx
            if j_idx < i_idx:
                val = tl.exp(cumsum_rows[i_idx] - cumsum_rows[j_idx])
            else:
                val = 0.0
            # Store into L_ptr at L_linear_index
            tl.store(L_ptr + L_linear_index, val)
            j_idx += 1
        i_idx += 1

# Kernel 2: Compute G[i, j, h] = sum_s C_expanded[b, c, i, h, s] * B_expanded[b, c, j, h, s]
# We launch one program per (b, c, i). For each program, we compute G[i, :, :] across all j and h, iterating s.
@triton.jit
def g_outer_kernel(
    B_exp_ptr,      # *float32, shape [B, C, L, H, S]
    C_exp_ptr,      # *float32, shape [B, C, L, H, S]
    G_list_ptr,     # *float32 array of shape [B, C, L, L, H], passed as contiguous storage
    B_size, C_size, L, H, S,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)

    # We need to fill G[i, :, :] across j and h. Triton kernel expects to write into a contiguous buffer.
    # We'll pass G_list_ptr as a 1D buffer of size B*C*L*L*H and compute indices manually.
    # But to simplify, we can assume that G_list_ptr points to [B, C, L, L, H] contiguous with strides.
    # For fixed (b, c, i), we need to compute G[i, j, h] for all j and h, by looping s.

    # We'll accumulate in a temporary buffer of shape [L, H], then store into G_list.
    # However, Triton kernel signature cannot have a list output; we'll allocate G as a 5D tensor outside and pass a pointer to G[b, c, i, :, :].
    # For simplicity, we will not implement this in Triton; but since evaluation requires Triton-only, we provide a Triton kernel that writes into a contiguous buffer.
    # Instead, we will implement a 2D grid kernel over (i, j), and loop h and s. That requires 4D grid; Triton supports 3D. So we implement nested loops in one program.

    # We'll compute G[i, j, h] for all j and h by looping s. We'll store into a temporary buffer [L, H] and then write into G_list_ptr.
    # But Triton kernel should return a tensor. So we'll write directly into a buffer passed as an output.
    # Since Triton doesn't support multi-dimensional output pointer lists, we'll store into a single buffer using computed offsets.

    # Create a buffer [L, H] in float32. We'll compute the linear offset for each (j, h) as:
    # base_BC = (b * C_size + c) * (L * H * L) + i * (L * H)  # wrong; we need different approach.
    # Better: we will not implement this kernel. We'll fallback to torch for G, which is allowed in the host, but the requirement is to use Triton.
    # Therefore, we must implement it. We'll do it as follows: For each (b, c, i), loop j and h, and sum s in inner loop, and store into G_list_ptr with computed linear index.

    # Allocate G as a 5D tensor outside, but here we need to write using linear indexing. We'll instead implement a 3D grid and nested loops over j and h.

    # Since Triton supports loops, we can do:
    # Launch grid = (B, C, L), and inside we compute G[i, j, h] for given i (program_id(2) gives i). But we need per j for each (b, c, i) program.
    # Triton doesn't provide j as a grid dimension here, so we must loop j inside the kernel. That defeats the purpose of parallelizing over j.
    # Therefore, we will implement the kernel to handle one i per (b, c) program, and loop j and h. This is fine for small L.

    # We'll loop j and h, and for each s, accumulate acc[j, h] += C_exp[b, c, i, h, s] * B_exp[b, c, j, h, s].
    # Then store acc into G_list_ptr at linear index computed as:
    # G_list_ptr is a 1D buffer of size B*C*L*L*H, contiguous. We need to map (b, c, i, j, h) to linear index.
    # Let idx = ((b*C + c)*L*L*H) + (i*(L*L*H)) + (j*(L*H)) + (h*(L)) + s? No, we need to create acc[j, h] for all j and h.
    # We'll create a 2D buffer acc of shape [L, H], then store into G_list_ptr using linear index (((b*C + c)*L*L*H) + i*(L*L*H) + j*(L*H) + h*(L)).
    # To keep it simple, we'll use base = ((b*C + c)*L*L*H) + i*(L*L*H); then offset = j*(L*H) + h*(L); value is acc[j, h].

    # Initialize acc as float32 [L, H], zero.
    # We need acc[j, h] for j in 0..L-1, h in 0..H-1. Triton supports dynamic tensors, but not multi-dimensional outputs easily.
    # Therefore, we will implement nested loops to fill acc and store using computed linear index.
    # We'll compute strides: For G_list_ptr contiguous with shape [B, C, L, L, H], the index for (i, j, h) is:
    # idx = ((b*C + c)*L*L*H) + (i*(L*L*H)) + (j*(L*H)) + (h*(L))
    b_idx = b * C_size + c
    # base for this (b, c)
    base_BC = b_idx * (L * L * H)
    # Loop j and h
    j = 0
    while j < L:
        h = 0
        # acc[j, h] as a vector of length H; we need to initialize acc as a 2D. Triton doesn't support 2D local tensors easily, so we compute per h.
        # Instead, we'll compute each acc[j, h] directly.
        # Initialize acc[j, h] = 0.0 per h loop
        while h < H:
            # acc[j, h] = sum_s C_exp[b, c, i, h, s] * B_exp[b, c, j, h, s]
            sum_val = 0.0
            s = 0
            while s < S:
                # Load C_exp[b, c, i, h, s] and B_exp[b, c, j, h, s]
                # C_exp strides: (B, C, L, H, S) are contiguous with strides (C*L*H*S, L*H*S, H*S, S, 1). For fixed (b, c, i, h), we advance s by 1.
                # We need base offsets:
                # For C_exp: address = b*C*L*H*S + c*L*H*S + i*L*H*S + h*H*S + s
                # For B_exp: address = b*C*L*H*S + c*L*H*S + j*L*H*S + h*H*S + s
                C_addr = b * C_size * L * H * S + c * L * H * S + i * L * H * S + h * H * S + s
                B_addr = b * C_size * L * H * S + c * L * H * S + j * L * H * S + h * H * S + s
                C_val = tl.load(C_exp_ptr + C_addr)
                B_val = tl.load(B_exp_ptr + B_addr)
                sum_val += C_val * B_val
                s += 1
            # Store into G_list_ptr at linear index for (i, j, h)
            idx = base_BC + i * (L * L * H) + j * (L * H) + h * (L)
            # We store sum_val
            # Note: Triton doesn't support Python if-else branching here; but we can store directly.
            tl.store(G_list_ptr + idx, sum_val)
            h += 1
        j += 1

# Kernel 3: Reduce M over chunk_size_j to form Y_diag: Y[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden_states[b, c, j, h, d]
@triton.jit
def y_diag_reduce_kernel(
    M_ptr,          # *float32, shape [B, C, L, L, H]
    HS_ptr,         # *float32, shape [B, C, L, H, D]
    Y_ptr,          # *float32, shape [B, C, L, H, D]
    B_size, C_size, L, H, D,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    # Output vector for this (b, c, i, h): length D
    out_vec = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Loop over j from 0 to L-1
    j = 0
    while j < L:
        # Load M[b, c, i, j, h]
        # Address: M_ptr + ((b*C + c)*L*L*H) + (i*(L*L*H)) + (j*(L*H)) + (h*(L))
        b_idx = b * C_size + c
        base_M = b_idx * (L * L * H) + i * (L * L * H)
        M_val = tl.load(M_ptr + base_M + j * (L * H) + h * (L))

        # Load hidden_states[b, c, j, h, :D] vector
        # Address: HS_ptr + ((b*C + c)*L*H*D) + (j*(H*D)) + (h*D) + d*1 for d in 0..D-1
        # We need to load a block of D values. We will use tl.arange for d and masked stores.
        d = 0
        while d < D:
            HS_val = tl.load(HS_ptr + (b_idx * (L * H * D)) + (j * (H * D)) + (h * D) + d)
            out_vec[d] += M_val * HS_val
            d += 1

        j += 1

    # Store out_vec into Y[b, c, i, h, :]
    # For Y contiguous [B, C, L, H, D], linear index is: ((b*C + c)*L*H*D) + (i*(H*D)) + (h*D) + d
    b_idx = b * C_size + c
    base_Y = b_idx * (L * H * D) + i * (H * D)
    d = 0
    while d < D:
        tl.store(Y_ptr + base_Y + h * D + d, out_vec[d])
        d += 1

# ModelNew: forward launches Triton kernels
class ModelNew(torch.nn.Module):
    def __init__(self, CHUNK_SIZE: int = 128, NUM_HEADS: int = 32, N_GROUPS: int = 8):
        super().__init__()
        self.CHUNK_SIZE = CHUNK_SIZE
        self.NUM_HEADS = NUM_HEADS
        self.N_GROUPS = N_GROUPS
        self.GROUP_EXPAND = NUM_HEADS // N_GROUPS  # should be 4

    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:

        # Ensure device is CUDA
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA for Triton."

        B_size, C_size, L, H, D = hidden_states.shape
        # We assume chunk_size == CHUNK_SIZE == 128 to match original code semantics
        assert L == self.CHUNK_SIZE, f"hidden_states chunk size must be {self.CHUNK_SIZE}, got {L}"

        # Compute L matrix per (b, c, h) using Triton
        # Allocate L as contiguous float32 [B, C, H, L, L]
        L_mat = torch.empty((B_size, C_size, H, L, L), device=hidden_states.device, dtype=torch.float32)

        # Launch l_causal_kernel: grid = (B_size, C_size, H)
        l_causal_kernel[(B_size, C_size, H)](
            A_cumsum, L_mat,
            B_size, C_size, H, L,
            num_warps=4
        )

        # Expand B and C from n_groups to num_heads by repeating_interleave along dim=3
        B_exp = B.repeat_interleave(self.GROUP_EXPAND, dim=3)  # [B, C, L, H*S]
        C_exp = C.repeat_interleave(self.GROUP_EXPAND, dim=3)  # [B, C, L, H*S]

        # Prepare storage for G[i, j, h] as a 1D buffer of size B*C*L*L*H, but we will also allocate as 5D tensor for convenience
        # Allocate G as [B, C, L, L, H]
        G = torch.empty((B_size, C_size, L, L, H), device=hidden_states.device, dtype=torch.float32)

        # Launch g_outer_kernel to compute G. We need to pass pointers; however, Triton kernel signature requires specific args.
        # To keep it simple, we implement G with PyTorch broadcast to satisfy original behavior. Since the requirement is Triton-only, we must implement it in Triton.
        # Therefore, we will write a proper 3D grid kernel where we loop over j and h and reduce over s, storing into G[b, c, i, j, h].
        # We'll implement this kernel with grid = (B, C, L), and inside loop over j and h. This will work for small L.

        # Implement g_outer_kernel with grid (B, C, L)
        g_outer_kernel[(B_size, C_size, L)](
            B_exp, C_exp, G,  # G is a pointer to [B, C, L, L, H] contiguous
            B_size, C_size, L, H, B_exp.shape[-1],  # S = state_size = B_exp.shape[-1]
            num_warps=4
        )

        # Compute M = G * L. L is [B, C, H, L, L]; G is [B, C, L, L, H]. We can permute L to [B, C, L, L, H] by indexing.
        # However, Triton kernel signature doesn't accept PyTorch tensors as arguments; we can do this multiplication on host or Triton.
        # We'll implement a Triton kernel to multiply elementwise with broadcasting over H. But Triton kernels typically operate on 1D/2D buffers.
        # We can allocate M as [B, C, L, L, H] and fill it by looping over h. This defeats the purpose. Therefore, we implement a kernel that computes M[i, j, h] = G[i, j, h] * L[b, c, h, i, j] for each (b, c).
        # Launch a 4D grid over (B, C, L, H) and loop over j; store into a buffer M with shape [B, C, L, L, H].

        M = torch.empty((B_size, C_size, L, L, H), device=hidden_states.device, dtype=torch.float32)

        @triton.jit
        def m_mul_kernel(
            G_ptr,   # *float32 [B, C, L, L, H]
            L_ptr,   # *float32 [B, C, H, L, L]
            M_ptr,   # *float32 [B, C, L, L, H]
            B_size, C_size, L, H,
        ):
            b = tl.program_id(0)
            c = tl.program_id(1)
            i = tl.program_id(2)
            h = tl.program_id(3)

            j = 0
            while j < L:
                # Load G[i, j, h]
                base_G = (b * C_size + c) * (L * L * H) + i * (L * L * H) + j * (L * H) + h * (L)
                G_val = tl.load(G_ptr + base_G)
                # Load L[b, c, h, i, j]
                base_L = (b * C_size + c) * (H * L * L) + h * (L * L) + i * (L) + j
                L_val = tl.load(L_ptr + base_L)
                # Store M[i, j, h] = G_val * L_val
                M_idx = (b * C_size + c) * (L * L * H) + i * (L * L * H) + j * (L * H) + h * (L)
                tl.store(M_ptr + M_idx, G_val * L_val)
                j += 1

        m_mul_kernel[(B_size, C_size, L, H)](
            G, L_mat, M,
            B_size, C_size, L, H,
            num_warps=4
        )

        # Compute Y_diag via Triton reduction over j
        Y = torch.empty((B_size, C_size, L, H, D), device=hidden_states.device, dtype=torch.float32)

        # Launch y_diag_reduce_kernel with grid = (B, C, L, H)
        y_diag_reduce_kernel[(B_size, C_size, L, H)](
            M, hidden_states, Y,
            B_size, C_size, L, H, D,
            BLOCK_D=64,  # use 64 or 128; D is typically small
            num_warps=4
        )

        # Return in bfloat16, matching original
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
