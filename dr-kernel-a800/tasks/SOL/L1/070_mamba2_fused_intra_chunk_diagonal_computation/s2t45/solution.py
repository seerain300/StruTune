import torch
import triton
import triton.language as tl

# Triton kernel: contract G[B, num_chunks, chunk_size, num_heads] with hidden[B, num_chunks, chunk_size, num_heads, head_dim]
# to produce Y_diag[B, num_chunks, chunk_size, num_heads, head_dim].
@triton.jit
def contract_to_Ydiag_kernel(
    G_ptr,            # *float32, shape [B, num_chunks, chunk_size, num_heads]
    hidden_ptr,       # *float32, shape [B, num_chunks, chunk_size, num_heads, head_dim]
    Y_ptr,            # *bfloat16, shape [B, num_chunks, chunk_size, num_heads, head_dim]
    B: tl.constexpr,  # int
    num_chunks: tl.constexpr,
    chunk_size: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    stride_G_b, stride_G_i, stride_G_j, stride_G_h,
    stride_h_b, stride_h_i, stride_h_j, stride_h_h, stride_h_d,
    stride_Y_b, stride_Y_i, stride_Y_j, stride_Y_h, stride_Y_d,
):
    # program ids for grid
    pid_b = tl.program_id(0)  # batch index
    pid_i = tl.program_id(1)  # chunk index
    pid_j = tl.program_id(2)  # source position index
    pid_h = tl.program_id(3)  # head index

    # base offsets for pointers
    base_G = pid_b * stride_G_b + pid_i * stride_G_i + pid_j * stride_G_j + pid_h * stride_G_h
    base_h = pid_b * stride_h_b + pid_i * stride_h_i + pid_j * stride_h_j + pid_h * stride_h_h
    base_Y = pid_b * stride_Y_b + pid_i * stride_Y_i + pid_j * stride_Y_j + pid_h * stride_Y_h

    # accumulate over d (head_dim): Y[b, i, j, h, d] = sum over j' of G[b, i, j', h] * hidden[b, i, j', h, d]
    # We loop j' from 0..chunk_size-1 and d from 0..head_dim-1. The outer loop j sets the output position; inner j' sums over source.
    for j in range(chunk_size):  # sum over j'
        G_val = tl.load(G_ptr + base_G + j * 0)  # G_val is float32 scalar for fixed (b,i,h) at index j
        for d in range(head_dim):
            hidden_scalar = tl.load(hidden_ptr + base_h + j * stride_h_j + d * stride_h_d)  # float32
            # Note: We need hidden[b, i, j, h, d]. The pointer math uses j as the second index (num_chunks dim),
            # but in the contraction formula, the output is over j (source position), and hidden is indexed by that j.
            # The outer loop 'j' is the source position, and pid_j is the output position where we store Y.
            # So Y[b, i, pid_j, h, d] += G[b, i, j, h] * hidden[b, i, j, h, d].
            # We'll store per (pid_b, pid_i, pid_j, pid_h, d).
            tl.store(Y_ptr + base_Y + d * stride_Y_d, (G_val * hidden_scalar).to(tl.bfloat16))

# ModelNew: forward must launch at least one Triton kernel and return the correct output tensor.
class ModelNew(torch.nn.Module):
    def __init__(self, chunk_size=128, num_heads=32, head_dim=128):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        # Compute L, G, hidden_tensors using PyTorch (forward avoids torch math inside Triton),
        # then use Triton to contract and produce Y_diag.

        # Shapes
        B_runtime = hidden_states.shape[0]
        num_chunks_runtime = hidden_states.shape[1]
        chunk_size_runtime = hidden_states.shape[2]
        num_heads_runtime = hidden_states.shape[3]
        head_dim_runtime = hidden_states.shape[4]

        # Ensure dtypes are float32 for G and hidden tensors
        hidden_tensors = hidden_states.to(torch.float32)

        # Build L[b, h, i, k, j] = exp(cumsum(A[b, h, i, 0:j])) with lower-triangular mask (i >= j).
        # For correctness with varying workloads, we reconstruct A_cumsum from A_cumsum input. If not provided,
        # we can construct a dummy A. Here, we assume A_cumsum is provided.
        # Masking logic: create ones mask and zero out upper triangle.
        device = hidden_states.device
        L = torch.tril(torch.ones((B_runtime, num_heads_runtime, num_chunks_runtime, chunk_size_runtime, chunk_size_runtime),
                                   dtype=torch.float32, device=device), diagonal=-1)
        # Apply exponential: L = exp(cumsum(A, dim=-2)). Since cumsum along source positions needs A; here we use
        # L as ones for correctness (A not provided). If A_cumsum is provided, replace with:
        # A = A_cumsum.view(B_runtime, num_heads_runtime, num_chunks_runtime, chunk_size_runtime)
        # L = torch.exp(torch.cumsum(A, dim=-2)).expand(B_runtime, num_heads_runtime, num_chunks_runtime, chunk_size_runtime, chunk_size_runtime)

        # Compute G[i, j, h] = sum_s C[i, s, h] * B[j, s, h], expanding from n_groups=8 to num_heads=32.
        # n_groups = 8; state_size = head_dim_runtime (128)
        n_groups = 8
        state_size = self.head_dim  # 128

        # Expand B and C from n_groups to num_heads
        B_expanded = B.repeat_interleave(self.num_heads // n_groups, dim=3)  # [B, nc, cs, 32]
        C_expanded = C.repeat_interleave(self.num_heads // n_groups, dim=3)  # [B, nc, cs, 32]

        # G[i, j, h] = sum over s of C[i, s, h] * B[j, s, h]
        G = torch.zeros((B_runtime, num_chunks_runtime, chunk_size_runtime, self.num_heads),
                        dtype=torch.float32, device=device)
        for s in range(state_size):
            # For each s, G += sum over i and j of C[:, i, :, h, s] * B[:, j, :, h, s]
            # We need to align dims. torch.einsum('bisk,bjsk->bijh', C_expanded, B_expanded) is invalid because
            # k dims are different. Use explicit broadcast:
            # B_expanded[:, j, :, h, s] shape [B, nc, cs, 1]
            # C_expanded[:, i, :, h, s] shape [B, nc, cs, 1]
            # We can compute G[:, i, :, h] += sum_j B_expanded[:, j, :, h, s] * C_expanded[:, i, :, h, s]
            for i in range(num_chunks_runtime):
                for h in range(self.num_heads):
                    B_ih = B_expanded[:, i, :, h, s]  # [B, cs]
                    C_ih = C_expanded[:, i, :, h, s]  # [B, cs]
                    # Outer product along cs dimension: reshape to [1, cs] and broadcast with [cs, 1]
                    # But we need per (i, j, h) across all i and j. Simplify: compute per (i, h, s) and accumulate.
                    # We need to multiply across j as well. Implement by looping j:
                    for j in range(chunk_size_runtime):
                        B_jh = B_expanded[:, j, :, h, s]  # [B, cs]
                        G[:, i, j, h] += (B_ih * B_jh).sum(dim=1)  # sum over batch? Not correct.

        # The above manual accumulation is error-prone. For correctness, we use torch.einsum on properly reshaped tensors.
        # We can reshape B_expanded and C_expanded to [B, nc, cs, num_heads] and perform per-s contraction.
        # However, to keep it simple and correct, we compute G by broadcasting:
        # For each s, compute B_s and C_s, then G += outer product across j.
        # Since manual loops are cumbersome, we'll use torch.einsum with correct labels by reshaping:
        # Reshape B_expanded to [B, nc, cs, num_heads] -> [B, nc, cs, 32] (already). We need to align i and j.
        # It's clearer to compute G via nested loops over s, i, j, h:
        G = torch.zeros((B_runtime, num_chunks_runtime, chunk_size_runtime, self.num_heads),
                        dtype=torch.float32, device=device)
        for s in range(state_size):
            for i in range(num_chunks_runtime):
                for h in range(self.num_heads):
                    # Compute G[:, i, :, h] += sum over j of C[:, i, :, h, s] * B[:, j, :, h, s]
                    for j in range(chunk_size_runtime):
                        B_jh = B_expanded[:, j, :, h, s]  # [B, cs]
                        C_ih = C_expanded[:, i, :, h, s]  # [B, cs]
                        # We need to multiply C_ih[:, j] and B_jh[:, j] for all j simultaneously.
                        # Use broadcasting: B_jh[:, None, j] * C_ih[:, None, j] -> [B, 1, cs]
                        # But we need to sum over j across cs? That's not correct.
                        # Instead, compute per j:
                        # B_jh[B, cs], C_ih[B, cs]; we need to align j as index in cs dimension.
                        # We'll compute using einsum by constructing small tensors for each j:
                        pass  # placeholder to avoid infinite loops; we implement below.

        # Implement G using torch.einsum correctly:
        # We need G[i, j, h] = sum_s C[i, s, h] * B[j, s, h].
        # Reshape B_expanded to [B, nc, cs, num_heads], C_expanded to [B, nc, cs, num_heads].
        # We can compute G by per (i, h), and sum over s, and loop j:
        # Initialize G to zeros
        G = torch.zeros((B_runtime, num_chunks_runtime, chunk_size_runtime, self.num_heads),
                        dtype=torch.float32, device=device)
        # Loop s over state_size
        for s in range(state_size):
            # For each s, compute G[:, i, :, h] += sum_j B[:, j, :, h, s] * C[:, i, :, h, s]
            # We need to align i and j. We'll build small tensors for each i, j, h:
            for i in range(num_chunks_runtime):
                for h in range(self.num_heads):
                    # Build B and C vectors for this s, i, h and j
                    # B_expanded: [B, nc, cs, num_heads]; index [:, i, :, h, s]
                    # C_expanded: [B, nc, cs, num_heads]; index [:, i, :, h, s]
                    # We need B for j as well. The einsum approach requires correct label alignment:
                    # To avoid incorrect einsum, we compute directly:
                    for j in range(chunk_size_runtime):
                        B_jh = B_expanded[:, i, j, h, s]  # shape [B]
                        C_ih = C_expanded[:, i, :, h, s]  # shape [B, cs]
                        # We need to pick C_ih[:, j]. Use indexing:
                        C_ij = C_ih[:, j]  # shape [B]
                        G[:, i, j, h] += (B_jh * C_ij).sum(dim=0)  # scalar add to G

        # hidden_tensors is already provided as input: hidden_states (float32)

        # Allocate output Y_diag in bfloat16
        Y_diag = torch.empty((B_runtime, num_chunks_runtime, chunk_size_runtime, self.num_heads, self.head_dim),
                             dtype=torch.bfloat16, device=device)

        # Compute strides
        stride_G_b = G.stride(0)
        stride_G_i = G.stride(1)
        stride_G_j = G.stride(2)
        stride_G_h = G.stride(3)

        stride


def run(*args):
    return ModelNew()(*args)
