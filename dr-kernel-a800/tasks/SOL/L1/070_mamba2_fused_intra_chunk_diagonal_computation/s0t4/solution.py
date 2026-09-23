import torch
import triton
import triton.language as tl

# Kernel A: Build L in flat buffer: L_flat[k], where k enumerates (b, c, i, j, h)
# L[i, j, h] = exp(sum_{k=0..j, k<=i} A[b, h, c, k]) for i >= j, else 0
@triton.jit
def build_L_flat_kernel(
    A_ptr,  # float32, shape [B, H, C, S]
    L_ptr,  # float32, shape [B*C*S*S*H]
    Bsz: tl.constexpr, Csz: tl.constexpr, H: tl.constexpr, S: tl.constexpr
):
    total = Bsz * Csz * S * S * H
    k = tl.program_id(0)
    # Compute (b, c, i, j, h) from k
    # k = (((((b*Csz + c)*S + i)*S + j)*H) + h)
    b = k // (Csz * S * S * H)
    tmp = k % (Csz * S * S * H)
    c = tmp // (S * S * H)
    tmp2 = tmp % (S * S * H)
    i = tmp2 // (S * H)
    tmp3 = tmp2 % (S * H)
    j = tmp3 // (H)
    h = tmp3 % (H)

    # Bounds check (shouldn't be needed if grid == total, but safe)
    if b >= Bsz or c >= Csz or i >= S or j >= S or h >= H:
        return

    # Map to A index: A[b, h, c, i] -> b*(H*C*S) + h*(C*S) + c*S + i
    A_idx = b * (H * Csz * S) + h * (Csz * S) + c * S + i
    a_val = tl.load(A_ptr + A_idx)  # scalar float32

    # Only include j <= i for lower-triangular
    # We need cumsum of A along j for each i, but Triton scalar loops are OK here with S=128.
    # Recompute cumsum for L[i, j] = exp(cum) if j <= i else 0
    # For each j, if j <= i: add a_val (since A_idx is fixed i). This is a vector of length S but we use scalar j-loop.
    # Better: for each j, if j <= i: L[b, c, i, j, h] = exp(sum_{kk=0..j, kk<=i} A[b, h, c, kk]).
    # Since we loop j in 0..S-1 (and k enumerates exactly one (i,j) per program), we can compute cum here.
    cum = tl.zeros((), dtype=tl.float32)  # scalar cumulative sum
    include = (j <= i)  # predicate whether to include j in i's row
    # Note: Triton supports scalar loops and conditionals; S is constexpr 128, so this is fine.
    # However, computing cum per j requires looping kk up to j. Triton doesn't allow vector range; we implement with scalar kk.
    # Initialize cumsum for L[i, j]: we need the sum of A along j up to j, but restricted to kk <= i.
    # Since for each kk we either include or not based on kk <= i, we can compute cum by iterating kk and adding a_val when kk <= i.
    # But a_val is A[i], not dependent on kk. The original L uses cumsum of A along j positions for each i. Since A is 1D per (b,h,c),
    # lower-triangular means for j>=i, we include only kk<=i. So cum at j is sum of A[b, h, c, kk] for kk in [0..min(i, S-1)].
    # We can compute this by iterating kk from 0 to S-1, adding A[b, h, c, kk] only if kk <= i (note kk <= i implies j >= kk, but we
    # must sum up to j; thus we need to include kk up to min(i, S-1)). For correctness, we implement cum as sum over kk<=i.
    # We obtain this by reloading A[b, h, c, kk] for kk<=i. This is acceptable given S=128 and H small.

    # Compute cum = sum_{kk=0..i} A[b, h, c, kk]
    cum = tl.zeros((), dtype=tl.float32)
    for kk in range(0, S):
        a_kk = tl.load(A_ptr + b * (H * Csz * S) + h * (Csz * S) + c * S + kk)
        # Only add if kk <= i
        if kk <= i:
            cum += a_kk
    # Now L[i, j] = exp(cum) if j <= i else 0
    L_val = tl.zeros((), dtype=tl.float32)
    if include:
        L_val = tl.exp(cum)
    # Store to flat L: L[b, c, i, j, h] linear index
    # k enumerates this index, so write L_val at L_ptr[k]
    tl.store(L_ptr + k, L_val)


# Kernel B: Compute G in flat buffer: G_flat[k] where k enumerates (b, c, i, j, h)
# G[i, j, h] = sum_n C_exp[b, c, i, n, j, h] * B_exp[b, c, j, n, i, h]
# Shapes:
#   C_exp: [B, C, S, H, N]
#   B_exp: [B, C, S, H, N]
@triton.jit
def compute_G_flat_kernel(
    B_exp_ptr,  # float32, shape [B, C, S, H, N]
    C_exp_ptr,  # float32, shape [B, C, S, H, N]
    G_ptr,      # float32, shape [B, C, S, S, H] -> flat
    Bsz: tl.constexpr, Csz: tl.constexpr, H: tl.constexpr, S: tl.constexpr, N: tl.constexpr
):
    total = Bsz * Csz * S * S * H
    k = tl.program_id(0)
    b = k // (Csz * S * S * H)
    tmp = k % (Csz * S * S * H)
    c = tmp // (S * S * H)
    tmp2 = tmp % (S * S * H)
    i = tmp2 // (S * H)
    tmp3 = tmp2 % (S * H)
    j = tmp3 // (H)
    h = tmp3 % (H)

    if b >= Bsz or c >= Csz or i >= S or j >= S or h >= H:
        return

    # Compute sum over n of C_exp[b, c, i, n, j, h] * B_exp[b, c, j, n, i, h]
    acc = tl.zeros((), dtype=tl.float32)
    # N is 128; loop over n
    for n in range(0, N):
        # B_exp[b, c, j, n, i, h] index:
        # ((b*Csz + c)*S + j)*H*N + n*H + h)*S + i -> linearize
        B_idx = (((b * Csz + c) * S + j) * H * N + n * H + h) * S + i
        B_val = tl.load(B_exp_ptr + B_idx)

        # C_exp[b, c, i, n, j, h] index:
        # ((b*Csz + c)*S + i)*H*N + n*H + h)*S + j
        C_idx = (((b * Csz + c) * S + i) * H * N + n * H + h) * S + j
        C_val = tl.load(C_exp_ptr + C_idx)

        acc += C_val * B_val

    tl.store(G_ptr + k, acc)


# Kernel C: Compute Y_diag in flat buffer: Y_flat[k] where k enumerates (b, c, i, h, d)
# Y[b, c, i, h, d] = sum_j G[b, c, i, j, h] * hidden_states[b, c, j, h, d]
@triton.jit
def compute_Y_diag_kernel(
    hidden_ptr,  # float32, shape [B, C, S, H, head_dim]
    G_ptr,       # float32, shape [B, C, S, S, H] -> flat
    Y_ptr,       # float32, shape [B*C*S*H*head_dim]
    Bsz: tl.constexpr, Csz: tl.constexpr, H: tl.constexpr, S: tl.constexpr, head_dim: tl.constexpr
):
    total = Bsz * Csz * S * H * head_dim
    k = tl.program_id(0)
    # Decode k into (b, c, i, h, d)
    b = k // (Csz * S * H * head_dim)
    tmp = k % (Csz * S * H * head_dim)
    c = tmp // (S * H * head_dim)
    tmp2 = tmp % (S * H * head_dim)
    i = tmp2 // (H * head_dim)
    tmp3 = tmp2 % (H * head_dim)
    h = tmp3 // head_dim
    d = tmp3 % head_dim

    if b >= Bsz or c >= Csz or i >= S or h >= H or d >= head_dim:
        return

    # Accumulate over j
    acc = tl.zeros((), dtype=tl.float32)
    for j in range(0, S):
        # G[b, c, i, j, h] -> flat index k = ((b*Csz + c)*S + i)*S*H + j*H + h
        G_idx = ((b * Csz + c) * S + i) * (S * H) + j * H + h
        G_val = tl.load(G_ptr + G_idx)

        # hidden[b, c, j, h, d] linear index:
        # (((b*Csz + c)*S + j)*H + h)*head_dim + d
        hidden_idx = (((b * Csz + c) * S + j) * H + h) * head_dim + d
        hidden_val = tl.load(hidden_ptr + hidden_idx)

        acc += G_val * hidden_val

    tl.store(Y_ptr + k, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Use the same constants as original context
        self.CHUNK_SIZE = 128    # S
        self.NUM_HEADS = 32      # H
        self.N_GROUPS = 8        # not used directly in kernels, but used to expand B/C in forward

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward. All heavy computation is done in Triton kernels.
        Inputs:
          - hidden_states: [B, C, S, H, head_dim]
          - A_cumsum:      [B, H, C, S]
          - B:             [B, C, S, N_GROUPS, N]
          - C:             [B, C, S, N_GROUPS, N]
        Output:
          - Y_diag:        [B, C, S, H, head_dim], dtype bfloat16
        """
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA for Triton."
        Bsz, Csz, S, H, head_dim = hidden_states.shape
        assert S == self.CHUNK_SIZE, f"hidden_states chunk size must be {self.CHUNK_SIZE}, got {S}"
        assert H == self.NUM_HEADS, f"num_heads must be {self.NUM_HEADS}, got {H}"

        # Ensure contiguity
        hidden_states = hidden_states.contiguous()
        A = A_cumsum.contiguous()  # [B, H, C, S]
        B = B.contiguous()
        C = C.contiguous()

        # Expand B and C along H to match NUM_HEADS=32 (repeat_interleave by 4)
        # Shapes after expand: [B, C, S, H, N]
        B_exp = B.repeat_interleave(self.NUM_HEADS // self.N_GROUPS, dim=3)
        C_exp = C.repeat_interleave(self.NUM_HEADS // self.N_GROUPS, dim=3)

        # Cast to float32 for computation
        A = A.to(torch.float32)
        B_exp = B_exp.to(torch.float32)
        C_exp = C_exp.to(torch.float32)
        hidden_states = hidden_states.to(torch.float32)

        # Allocate flat buffers (float32 for numerical stability)
        total_L = Bsz * Csz * S * S * H
        L_flat = torch.empty(total_L, dtype=torch.float32, device=hidden_states.device)

        total_G = Bsz * Csz * S * S * H
        G_flat = torch.empty(total_G, dtype=torch.float32, device=hidden_states.device)

        total_Y = Bsz * Csz * S * H * head_dim
        Y_flat = torch.empty(total_Y, dtype=torch.float32, device=hidden_states.device)

        # Launch kernel A: build L_flat
        grid_L = (total_L,)
        build_L_flat_kernel[grid_L](A, L_flat, Bsz=Bsz, Csz=Csz, H=H, S=S)

        # Launch kernel B: compute G_flat
        N = C_exp.size(-1)  # state_size; typically 128
        grid_G = (total_G,)
        compute_G_flat_kernel[grid_G](B_exp, C_exp, G_flat, Bsz=Bsz, Csz=Csz, H=H, S=S, N=N)

        # Launch kernel C: compute Y_flat
        grid_Y = (total_Y,)
        compute_Y_diag_kernel[grid_Y](hidden_states, G_flat, Y_flat, Bsz=Bsz, Csz=Csz, H=H, S=S, head_dim=head_dim)

        # Reshape and cast to bfloat16 to match original return type
        Y = Y_flat.view(Bsz, Csz, S, H, head_dim)
        Y = Y.to(torch.bfloat16)
        return Y


def run(*args):
    return ModelNew()(*args)
