import torch
import triton
import triton.language as tl


# Triton Kernel: Compute masked cumsum along source index (dim=-2) for each (b, n, c),
# then apply exp to get causal mask L. We implement lower-triangular mask with diagonal=-1:
# include j < i (exclude j == i). Output L: [B, C, S, S, N] in float32.
@triton.jit
def masked_cumsum_exp_lower_kernel(
    A_ptr, L_ptr,
    B_size, C_size, N,
    S: tl.constexpr,  # chunk_size
    num_warps=1, num_stages=1
):
    # program ids
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_n = tl.program_id(2)  # head index

    # running sum vector across source indices
    running = tl.zeros([S], dtype=tl.float32)

    # Loop over target index i from 0 to S-1
    for i in tl.static_range(S):
        total = tl.zeros((), dtype=tl.float32)
        # For each source index j, if j < i then include; else zero
        for j in tl.static_range(S):
            # Compute linear offset for A[b, n, c, i, j]
            # A is [B, N, C, S, S] but we pass it as flattened; we use strides to index.
            # However, to keep it simple and match original axes, we assume A_ptr points to
            # a tensor [B, N, C, S, S] contiguous. We can decode indices with strides.
            # Here, we use flattened indexing: idx = ((b*N + n)*C + c)*S + i*S + j
            # But we need actual strides. We'll assume A_ptr is laid out as [B,N,C,S,S] and
            # use element strides. Since Triton doesn't expose stride via tl.load, we compute
            # index manually using S as element stride. To make this robust, we pass A as
            # contiguous [B,N,C,S,S] and compute index accordingly.
            # Compute offset for A: ((b*N + n)*C + c)*S + i*S + j
            # Note: A pointer is [B,N,C,S,S]; we decode as idx = b*(N*C*S*S) + n*(C*S*S) + c*(S*S) + i*S + j.
            # But passing pointer directly is better: we pass A with shape (B,N,C,S,S).
            # We'll compute offset using S as stride in last dim: index = ((b*N + n)*C + c)*S*S + i*S + j
            # However, Triton can't use runtime sizes for indexing; we assume A_ptr is [B,N,C,S,S] contiguous.
            # Use flattened indexing with 5D grid: index = b*(N*C*S*S) + n*(C*S*S) + c*(S*S) + i*S + j.
            # To keep it simple, we pass A_ptr as [B,N,C,S,S] and index via tl.arange.
            # We'll create vector offsets for j across S.
            # For simplicity, we compute element-wise access using offsets: idx = ((b*N + n)*C + c)*S + i*S + j
            # Here, we use 5D strides by passing pointers with shape. Triton will decode offsets from tl.arange.
            # Better: use tl.load with pointer arithmetic: ptr = A_ptr + b*stride_b + n*stride_n + c*stride_c + i*stride_i + j*stride_j
            # But since we only have one pointer, we compute index as linear: idx = ((b*N + n)*C + c)*S*S + i*S + j.
            # Triton supports element indexing via pointer + offset. We'll compute offset with tl.arange on S.
            # Create a vector of j offsets for this i and load A[b,n,c,i,j] with mask j < i.
            # Since we don't have explicit strides, we assume A_ptr is contiguous [B,N,C,S,S].
            # We'll launch kernel with grid (B, C, N) and compute index as above.
            # Compute index = ((b*N + n)*C + c)*S*S + i*S + j
            # Triton allows scalar arithmetic; we'll use a loop per i.

            # Instead of computing index, we directly use tl.load with pointer offset.
            # We need to pass A_ptr with shape [B,N,C,S,S]. Triton can't infer strides, so we index linearly.
            # For each i and j, compute idx = ((b*N + n)*C + c)*S*S + i*S + j, and load A[b,n,c,i,j].
            # This assumes A is contiguous [B,N,C,S,S]. We'll ensure A tensor is contiguous before launch.

            # To do this, we need to compute the address. Triton doesn't expose .shape, so we pass only pointer.
            # We'll emulate by using flattened indexing: index = ((pid_b*N + pid_n)*C + pid_c)*S*S + i*S + j
            # But in kernel we don't have pid_b; we need to pass B_size as b. We'll use tl.program_id(0) corresponds to b.
            b = tl.program_id(0)
            n = tl.program_id(2)
            c = tl.program_id(1)

            # Compute index = ((b*N + n)*C + c)*S*S + i*S + j
            # Convert N, C, S to scalars for arithmetic: assume N, C, S are passed as tl.constexpr? No, they are runtime.
            # Triton requires constexpr for tl.static_range and some meta-params. We pass S as constexpr, N and C as runtime.
            # We can use runtime arithmetic: total = running[j] for j < i. Since running is vector, we sum masked entries.
            # Let's implement masked addition using running vector.

            # We need A[b, n, c, i, j] values for j in [0..S-1]. We'll construct offsets vector for j and load.
            # Triton supports vector loads via tl.arange. We'll load vector of j values.
            j_vec = tl.arange(0, S)
            mask_j = j_vec < i  # j < i
            # Compute offsets vector for A: ((b*N + n)*C + c)*S + i*S + j_vec
            # Since we don't have explicit strides, we assume A_ptr points to a contiguous [B,N,C,S,S] tensor.
            # We'll compute base = ((b*N + n)*C + c)*S*S; then index = base + i*S + j_vec.
            base = ((b * N + n) * C + c) * S * S
            offset_vec = base + i * S + j_vec
            # Load values with mask
            A_vals = tl.load(A_ptr + offset_vec, mask=mask_j, other=0.0)
            # Compute cumsum of A_vals into total
            # We need to sum A[b, n, c, i, j] for j < i. total += A_vals[mask_j]
            # Since A_vals has S elements, we sum only masked ones.
            # Triton doesn't have vectorized sum reduction, so we compute manually:
            for jj in tl.static_range(S):
                include = jj < i
                total += A_vals[jj] * include  # include is bool; Triton will handle type

            # running update: running[j] = (j < i) ? total : 0
            # We store L[b, n, c, i, j] = exp(total) for j < i; else 0.
            # For j >= i, value is 0 (due to mask). We'll store exp(total) with mask.
            # We need to compute L index: ((b*N + n)*C + c)*S + i*S + j
            L_idx = base + i * S + j_vec
            L_vals = tl.exp(tl.full([S], total, dtype=tl.float32)) * mask_j
            tl.store(L_ptr + L_idx, L_vals, mask=mask_j)

# Triton Kernel: Contract B_expanded and C_expanded to G[b, c, i, j, n] = sum_k C[b, c, i, n, k] * B[b, c, j, n, k]
@triton.jit
def contract_BC_to_G_kernel(
    B_ptr, C_ptr, G_ptr,
    B_size, C_size, N,
    S: tl.constexpr,  # chunk_size
    K: tl.constexpr,  # state_size = hidden_states.last_dim
    num_warps=1, num_stages=1
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)
    # Loop over state_size K
    for k in tl.static_range(K):
        # Load B[b, c, j, n, k] and C[b, c, i, n, k]
        # Assume B_ptr, C_ptr are [B, C, S, N, K] contiguous tensors.
        # B index: ((b*C + c)*S + j)*N*K + n*K + k
        # C index: ((b*C + c)*S + i)*N*K + n*K + k
        B_idx = ((b * C_size + c) * S + j) * N * K + n * K + k
        C_idx = ((b * C_size + c) * S + i) * N * K + n * K + k
        B_val = tl.load(B_ptr + B_idx)
        C_val = tl.load(C_ptr + C_idx)
        acc += B_val * C_val

    # Store G[b, c, i, j, n] = acc
    G_idx = ((b * C_size + c) * S + i) * S * N + j * N + n
    tl.store(G_ptr + G_idx, acc)


# Triton Kernel: Diagonal contraction to compute Y[b, c, i, n, d] = sum_j M[b, c, i, j, n] * hidden_states[b, c, j, n, d]
@triton.jit
def diag_contract_Y_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    B_size, C_size, N,
    S: tl.constexpr,  # chunk_size
    D: tl.constexpr,  # head_dim
    num_warps=1, num_stages=1
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    n = tl.program_id(3)
    d = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)
    for j in tl.static_range(S):
        # Load M[b, c, i, j, n]
        M_idx = ((b * C_size + c) * S + i) * S * N + j * N + n
        M_val = tl.load(M_ptr + M_idx)
        # Load hidden_states[b, c, j, n, d]
        # hidden_ptr is [B, C, S, N, D] contiguous
        hidden_idx = ((b * C_size + c) * S + j) * N * D + n * D + d
        hidden_val = tl.load(hidden_ptr + hidden_idx)
        acc += M_val * hidden_val

    # Store Y[b, c, i, n, d] = acc
    Y_idx = ((b * C_size + c) * S + i) * N * D + n * D + d
    tl.store(Y_ptr + Y_idx, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants from original signature
        self.NUM_HEADS = 32
        self.N_GROUPS = 8

    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [batch_size, num_chunks, chunk_size, num_heads, head_dim]
        A_cumsum: [batch_size, num_heads, num_chunks, chunk_size, chunk_size]
        B: [batch_size, num_chunks, chunk_size, n_groups, state_size]
        C: [batch_size, num_chunks, chunk_size, n_groups, state_size]
        Output: [batch_size, num_chunks, chunk_size, num_heads, head_dim], dtype bfloat16
        """
        device = hidden_states.device
        Bsz, Csz, S, N, D = hidden_states.shape  # infer axes

        # Ensure tensors are contiguous
        A = A_cumsum.contiguous()  # [Bsz, N, Csz, S, S]
        B_ = B.contiguous()        # [Bsz, Csz, S, self.N_GROUPS, D]
        C_ = C.contiguous()        # [Bsz, Csz, S, self.N_GROUPS, D]

        # 1) Compute L in Triton: masked cumsum with diagonal=-1, then exp
        # L: [Bsz, Csz, S, S, N] float32
        L = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)

        # Launch Triton kernel: grid = (B, C, N)
        grid_L = (Bsz, Csz, N)
        masked_cumsum_exp_lower_kernel[grid_L](
            A, L,
            B_size=Bsz, C_size=Csz, N=N,
            S=S,  # chunk_size as constexpr meta-param
            num_warps=1, num_stages=1
        )

        # 2) Expand B and C to num_heads (repeat_interleave by 4)
        # B_expanded: [Bsz, Csz, S, N, D]
        # C_expanded: [Bsz, Csz, S, N, D]
        # Note: In original, NUM_HEADS // N_GROUPS = 4, so repeat_interleave(4).
        # We implement repeat_interleave along dim=3.
        # However, Triton kernels here expect shapes [B, C, S, N, K]; we can create expanded tensors in PyTorch.
        # For performance, we can keep as PyTorch repeat_interleave; it's a light op.
        repeat_factor = self.NUM_HEADS // self.N_GROUPS  # 4
        B_expanded = B_.repeat_interleave(repeat_factor, dim=3)  # [Bsz, Csz, S, N, D]
        C_expanded = C_.repeat_interleave(repeat_factor, dim=3)  # [Bsz, Csz, S, N, D]

        # 3) Compute G via Triton contraction over state_size K=D
        G = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)
        grid_G = (Bsz, Csz, S, S, N)
        contract_BC_to_G_kernel[grid_G](
            B_expanded, C_expanded, G,
            B_size=Bsz, C_size=Csz, N=N,
            S=S, K=D,
            num_warps=1, num_stages=1
        )

        # 4) M = G * L (elementwise). We keep this as PyTorch multiply (not forbidden here).
        M = G * L  # float32

        # 5) Diagonal contraction to compute Y_diag: [Bsz, Csz, S, N, D], float32
        Y = torch.empty((Bsz, Csz, S, N, D), device=device, dtype=torch.float32)

        hidden_expanded = hidden_states  # already [Bsz, Csz, S, N, D] logically; we pass as-is
        # Ensure hidden_expanded has last dim D: it does.

        grid_Y = (Bsz, Csz, S, N, D)
        diag_contract_Y_kernel[grid_Y](
            M, hidden_expanded, Y,
            B_size=Bsz, C_size=Csz, N=N,
            S=S, D=D,
            num_warps=1, num_stages=1
        )

        # Cast to bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
