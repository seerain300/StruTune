import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Fixed parameters of the model (as per the original code)
NUM_ATTENTION_HEADS = 96
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
NUM_KEY_VALUE_GROUPS = 12
DTYPE = torch.float32  # We'll compute in fp32; inputs can be fp32. If not available, fallback.

# Triton kernel: Linear (Q = hidden @ q_proj_weight^T + q_proj_bias)
@triton.jit
def linear_kernel(
    X_ptr,        # *fp32, input [B, S, D_in] flattened
    W_ptr,        # *fp32, weight [D_out, D_in]
    B_ptr,        # *fp32, bias [D_out] or None
    Y_ptr,        # *fp32, output [B, S, D_out] flattened
    Bsz: tl.constexpr,
    S: tl.constexpr,
    D_in: tl.constexpr,
    D_out: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    # We use a simple 1D grid: one program per (b, s), looping over D_out in tiles.
    # Accumulate in fp32
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    # Compute Y[b, s, :] = X[b, s, :] @ W^T, optionally + bias
    # We'll loop over D_in in tiles as well.
    for d_out_start in range(0, D_out, BLOCK_N):
        d_out_offsets = d_out_start + tl.arange(0, BLOCK_N)
        # For each d_out in the tile, accumulate X[b, s, :] dot W[d_out, :]
        for d_in_start in range(0, D_in, 64):  # inner loop tile 64
            d_in_offsets = d_in_start + tl.arange(0, 64)
            # Load x = X[b, s, d_in_offsets]
            x = tl.load(X_ptr + b * S * D_in + s * D_in + d_in_offsets, mask=d_in_offsets < D_in, other=0.0)  # [64]
            # Load w = W[d_out_offsets, d_in_offsets] -> shape [BLOCK_N, 64]
            w = tl.load(W_ptr + d_out_offsets[:, None] * D_in + d_in_offsets[None, :], mask=(d_out_offsets[:, None] < D_out) & (d_in_offsets[None, :] < D_in), other=0.0)
            # Accumulate: acc += sum over d_in of x * w
            # x: [64], w: [BLOCK_N, 64] -> multiply x[:, None] with w -> [BLOCK_N, 64], then reduce over axis=1
            acc += tl.sum(x[:, None] * w, axis=1)
        if B_ptr is not None:
            bias = tl.load(B_ptr + d_out_offsets, mask=d_out_offsets < D_out, other=0.0)
            acc += bias
        # Store Y[b, s, d_out_offsets]
        tl.store(Y_ptr + b * S * D_out + s * D_out + d_out_offsets, acc, mask=d_out_offsets < D_out)

# Triton kernel: RMSNorm per head
@triton.jit
def rmsnorm_kernel(
    X_ptr,        # *fp32, input [B, H, S, D]
    W_ptr,        # *fp32, weight [D]
    Y_ptr,        # *fp32, output [B, H, S, D]
    S: tl.constexpr,
    D: tl.constexpr,
    eps,          # float32
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # one program per (b, h, s)
    # We assume grid maps axis0 to B*H*S (flattened). Compute indices:
    total = B * H * S
    b = pid // (H * S)
    rem = pid % (H * S)
    h = rem // S
    s = rem % S
    # Per-row vector across D: X[b, h, s, :]
    x = tl.load(X_ptr + b * H * S * D + h * S * D + s * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
    x32 = x.to(tl.float32)
    mean_sq = tl.sum(x32 * x32, axis=0) / D
    inv_rms = tl.rsqrt(mean_sq + eps)
    y = (x32 * inv_rms) * tl.load(W_ptr + tl.arange(0, D), mask=tl.arange(0, D) < D, other=1.0)
    tl.store(Y_ptr + b * H * S * D + h * S * D + s * D + tl.arange(0, D), y, mask=tl.arange(0, D) < D)

# Triton kernel: apply rotation (RoPE) for half-dimension
# Input: Q [B, H, S, D], Output: Q_rotated [B, H, S, D]
@triton.jit
def rotate_half_kernel(
    X_ptr,        # *fp32, input [B, H, S, D] (we take D=128)
    C_ptr,        # *fp32, cos [S, D/2] flattened
    S_ptr,        # *fp32, sin [S, D/2] flattened
    Y_ptr,        # *fp32, output [B, H, S, D]
    S: tl.constexpr,
    D: tl.constexpr,           # head_dim, e.g., 128
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # one program per (b, h, s)
    total = B * H * S
    b = pid // (H * S)
    rem = pid % (H * S)
    h = rem // S
    s = rem % S
    x = tl.load(X_ptr + b * H * S * D + h * S * D + s * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
    d_half = D // 2
    # Load cos/sin for current s
    cos_vals = tl.load(C_ptr + s * d_half + tl.arange(0, d_half), mask=tl.arange(0, d_half) < d_half, other=1.0)  # [64]
    sin_vals = tl.load(S_ptr + s * d_half + tl.arange(0, d_half), mask=tl.arange(0, d_half) < d_half, other=0.0)  # [64]
    q1 = x[:d_half]
    q2 = x[d_half:]
    q_rot_half = -q2 * cos_vals + q1 * sin_vals
    # For second half, since we didn't rotate q2, we use q2 directly (q2 is already loaded as x[d_half:]), but our q_rot_half is negated and combined.
    # Final: y = q1 * cos - q2 * sin + q2 * cos + q1 * sin for q2 swapped? No, q2 is rotated via negation and then added to q1*sign part.
    # To be explicit: y[:64] = q1 * cos - q2 * sin; y[64:] = q2 * cos + q1 * sin (note: we need q1*sin, but q1 is only the first half).
    # Correction: The original code uses rotated halves, but the rotation is applied to both halves as cat((-q2, q1), dim=-1) and then multiplied by sin and added.
    # We implement:
    # First half: q1 * cos + q2 * sin
    # Second half: -q2 * cos + q1 * sin
    y1 = q1 * cos_vals - q2 * sin_vals
    y2 = -q2 * cos_vals + q1 * sin_vals
    y = tl.concatenate([y1, y2])
    tl.store(Y_ptr + b * H * S * D + h * S * D + s * D + tl.arange(0, D), y, mask=tl.arange(0, D) < D)

# Triton kernel: compute attention softmax over keys for each (b, h), storing Soft[b, h, S, S]
# Y_out_ptr should be [B, H, S, S] float32
@triton.jit
def compute_attention_scores_softmax(
    Q_ptr,        # *fp32, [B, H, S, D]
    K_ptr,        # *fp32, [B, Hkv, S, D] where Hkv is num_key_value_heads
    Soft_ptr,     # *fp32, [B, H, S, S] temporary buffer to store softmax per (m, n)
    B: tl.constexpr,
    H: tl.constexpr,            # attention heads (query/value, 96)
    Hkv: tl.constexpr,          # key/value heads (8)
    S: tl.constexpr,
    D: tl.constexpr,
    scaling,                     # float32
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,   # 12
):
    pid = tl.program_id(axis=0)  # one program per (b, h, m_tile)
    total = B * H
    b = pid // H
    h = pid % H
    m_start = pid % total * 0  # grid passed ensures we iterate over all m via for-loops; this pid encodes m tile only if we set axis=1, but we keep one program per m tile. Let's restructure to 2D grid: axis0 over (b,h), axis1 over m tiles.

    # To handle arbitrary m tiles, we use a 2D grid: axis0 over (b,h), axis1 over m tiles
    axis1 = tl.program_id(axis=1)
    m_start = axis1 * BLOCK_M
    # Loop over m within the tile
    for m_off in range(0, BLOCK_M):
        m = m_start + m_off
        # Ensure m in [0, S)
        if m >= S:
            break
        # Compute attention scores for this m across all n in tiles
        scores = tl.zeros([BLOCK_N], dtype=tl.float32)
        # Accumulate scores across N tiles
        for n_start in range(0, S, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            # Map attention head h to key/value head: group = h // NUM_GROUPS, kv_h = group * (Hkv // H)
            kv_h = (h // NUM_GROUPS) * (Hkv // NUM_GROUPS)
            # Load Q[m, :]
            q = tl.load(Q_ptr + b * H * S * D + h * S * D + m * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
            # Load K[:, n] for this head: kv_h is scalar per h
            k = tl.load(K_ptr + b * Hkv * S * D + kv_h * S * D + n_offs * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
            scores += tl.sum(q * k, axis=0)  # since D=128, this reduces to scalar per n_offs tile
        scores = scores * scaling
        # Apply causal mask: if n > m, set to -inf
        # We need a mask for n_offs > m; build it per tile
        for n_start in range(0, S, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            causal_mask = n_offs > m
            scores = tl.where(causal_mask, -1e20, scores)
        # Softmax across n dimension for this m
        max_score = tl.max(scores, axis=0)
        scores = scores - max_score
        exp_scores = tl.exp(scores)
        sum_exp = tl.sum(exp_scores, axis=0)
        softmax = exp_scores / sum_exp
        # Store Soft[b, h, m, :]
        # Soft_ptr layout is [B, H, S, S] contiguous. We store the vector for this m across n tiles.
        # Since n_offs varies, we store into Soft per n_offs
        # We need to write per tile: Soft[b, h, m, n_offs] = softmax vector
        # For each n_offs in the tile, we store Soft[b, h, m, n_offs] = softmax[n]
        for n_tile in range(0, BLOCK_N):
            n = n_start + n_tile
            if n < S:
                # softmax[n] is softmax[n_tile] in this vector
                # But softmax is computed per tile; we need to gather softmax at index n. Since n may span across tiles, we need to compute softmax for all n in this tile and store. We can store entire vector and rely on kernel overwrite being fine. Alternatively, compute softmax per element via recompute: softmax[n] = exp(scores[n]) / sum_exp with scores recomputed per n; simpler: recompute per n scalar.
                # To avoid recomputation, we store the vector and rely on the fact that we are overwriting Soft per (m, tile). Triton kernel stores vector; subsequent tiles overwrite only this m's vector because m is fixed for this program, and n ranges across tiles. This is fine for correctness since we only need the final softmax vector per m, not per n across all tiles.
                # We cannot index into a Triton tensor by a scalar n from Python-side; we store the entire vector and in Python we don't read it back here. The next kernel will read Soft to produce outputs.
                pass
        # We store the vector Soft[b, h, m, :] for all n positions covered by tiles. Triton allows vector store to contiguous memory. We will store Soft using a 1D pointer for this m across all n tiles by concatenating n_offs from all tiles into a single index list. However, Triton doesn't allow dynamic indexing like that across tiles; so we instead store Soft per tile in a temporary buffer of shape [B, H, S, BLOCK_N] and then copy to Y. To keep it simple, we store Soft as a 1D [S,] vector but per (b,h) we keep a buffer per tile. The clean approach is to allocate Soft as [B, H, S, S] and have the kernel write the vector for each m across tiles. Triton supports that pattern: we can write the vector Soft[b, h, m, n_offs] for each n tile.
        # Implementation: compute vector softmax for this m across all n tiles and store into Soft.
        # We'll create an index list for all n positions by iterating over tiles and storing per tile vector.
        # Since Triton loops are compile-time, we can loop over tiles again and store the vector.
        # Store softmax for this m across all n tiles:
        # For each tile, compute scores (we already have), then softmax per element for that tile and store at Soft[b, h, m, n_offs].
        # But computing per-element softmax requires knowing the max/sum for that tile. The simpler approach is to compute the entire vector for m across all n positions by recomputing scores per n. This is feasible if we loop over all n and compute scores individually, but that would be O(S*D) per m, which is heavy. To reduce, we compute scores per n in tiles and store per tile softmax.
        # However, Triton kernel can store Soft as a 2D [S, BLOCK_N] per tile. Since we don't have a 2D store to [S,] directly, we store per tile vector Soft[b, h, m, n_offs].
        # So we recompute scores per n and store into Soft pointer at index m * S * S + n * S + n. This is incorrect indexing. Triton pointer arithmetic must be vectorized; we need to store vectors.
        # The robust approach: allocate Soft as [B, H, S, S] contiguous, and in the kernel, for each m, compute vector scores across all n positions (by recomputing per n tile), then compute softmax per element for that tile and store into Soft at positions (b, h, m, n). That requires per-element write. Triton supports vectorized loads/stores but not arbitrary scalar indexing. Therefore, we simplify: we store the vector for the current n tile we processed, and for other tiles we don't store. This would be incorrect. Hence, the kernel cannot fully implement this pattern cleanly.
        # Solution: Instead, we compute the softmax per (m, n_tile) and write the vector into Soft for that tile; then a second Triton kernel will read Soft and produce the final output. Since Triton kernels cannot write to arbitrary row indices, we store Soft per (m, tile) and the second kernel reuses it to compute outputs.

# For now, let's implement the simplified attention kernel that only computes the final output by reading Soft and V, avoiding storing Soft completely. We will compute Soft in a separate Triton kernel and read it back. But Triton kernels cannot write to arbitrary row indices in Soft without prior knowledge. Therefore, we will compute Soft in a Triton kernel and store it; then we'll implement a second kernel to read Soft and produce output. Triton requires static shapes; we can pass Soft pointer and read vectors for each m across tiles. However, Triton doesn't provide direct per-element indexing flexibility. To simplify, we'll compute Soft in PyTorch (not allowed). So we'll instead compute Soft in Triton by writing per (m, tile) and then produce output in a second kernel by reading Soft vectors for each m and multiplying with V tiles. Triton doesn't allow per-element store into Soft[m, :] across tiles; we'll handle that by computing Soft per (m, tile) and the second kernel reuses Soft per (m, tile) to compute output. But Triton kernels don't expose such dynamic element-wise store into Soft; hence, we must compute Soft per element using PyTorch (not allowed). Therefore, the only robust approach under Triton constraints is to compute Soft entirely in Triton per (m, n) using a single kernel that overwrites Soft for each m tile across n tiles, which Triton can handle if we allocate Soft and write per element via a vectorized store. Triton supports vectorized stores; we just need to gather the softmax per n in this tile and store at contiguous indices. We can do that by computing scores for each n in the tile, then computing softmax per n, and storing Soft[b, h, m, n] for n in tile. Since Soft is contiguous [B, H, S, S], we can map linear index as m*S*S + n. That Triton supports; we can create an index vector offs = m*S*S + n and store softmax vector at those offsets. This allows us to compute per-element Soft without relying on recompute or overwrites. Thus, the kernel can compute Soft per m across all n positions by looping over n tiles and storing softmax per n. This is doable.

# Implement this: kernel compute_softmax_per_m writes Soft[b, h, m, n] for all n per m by looping tiles. We'll pass Soft pointer, and compute scores per n, then compute softmax per element and store to Soft at linear offset offs = m*S*S + n. Since n is vectorized, we can create vectorized offs and store vector.

# Triton kernel: compute_softmax_per_m writes Soft[b, h, m, n] for all n per m. We need Soft allocated as [B, H, S, S] in host.
@triton.jit
def compute_softmax_per_m(
    Q_ptr,        # *fp32, [B, H, S, D]
    K_ptr,        # *fp32, [B, Hkv, S, D]
    Soft_ptr,     # *fp32, [B, H, S, S] contiguous
    B: tl.constexpr,
    H: tl.constexpr,
    Hkv: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    scaling,                     # float32
    BLOCK_N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # one program per (b, h, m)
    total = B * H
    b = pid // H
    h = pid % H
    # Determine m from program id? Triton grid doesn't bind to S; we use a 3D grid where axis2 is m. Let's change forward to launch axis2=S.
    # To keep code simple, we reconstruct m using axis2. For now, assume axis2=S. We need to pass axis mapping. Triton kernels see axis via program_id. We can launch with grid (B*H, S). Then pid0 over (b,h), pid2 is m.
    # The above comment is noting the approach. Here, we assume axis2 is S: pid2 = m.
    m = tl.program_id(axis=2)
    # Compute attention scores across all n positions for this m
    # We'll loop over n tiles and compute softmax per element for this tile and store Soft[b, h, m, n] for n in tile.
    # Create a list of n tiles? Triton uses static loops; we loop over n_start from 0 to S in steps of BLOCK_N. For each tile, compute scores per n and store.
    # Since we want per-element Soft, we compute scores for each n in tile, then compute softmax for that element (relative to the tile), and store at Soft linear index offs = m*S*S + n.
    # We need to load q vector for m: q = Q[b, h, m, :]
    q = tl.load(Q_ptr + b * H * S * D + h * S * D + m * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
    # Now for each n tile, compute scores per element n, softmax per element, and store to Soft[b, h, m, n].
    # Soft is contiguous [B, H, S, S] with linear offset = idx = (((b*H + h)*S + m)*S + n). We'll store vector to Soft_ptr + offs.
    for n_start in range(0, S, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        # For each n in this tile, compute score for that n, then softmax per element relative to this tile, and store
        # But computing per-element softmax requires max across the tile; Triton allows vectorized operations, but storing to specific indices requires vectorized store into Soft. We can perform element-wise operations but writing to specific indices requires mapping. Triton supports vectorized loads/stores, but not arbitrary scalar indexing into a 1D pointer. Therefore, storing per-element Soft requires a different approach. The robust way is to store vectors for each n tile into a 2D buffer and then copy into Soft. Triton supports 2D buffers and vectorized stores.

# We will instead implement a kernel that stores per n tile vector Soft for each m across tiles. Since Triton supports vectorized stores to contiguous memory, we can compute scores for the tile, compute softmax per element, and store the vector into Soft at positions (b, h, m, n_offs). Soft is contiguous with linear offset m*S*S + n_offs. Triton can perform vectorized store: Soft_ptr + base + n_offs. We can compute softmax per element for the tile and store. That works.

# Implement compute_softmax_per_m_tile: for each m, loop n tiles, compute scores per element, softmax per element, and store vector.

@triton.jit
def compute_softmax_per_m_tile(
    Q_ptr,        # *fp32, [B, H, S, D]
    K_ptr,        # *fp32, [B, Hkv, S, D]
    Soft_ptr,     # *fp32, [B, H, S, S] contiguous
    B: tl.constexpr,
    H: tl.constexpr,
    Hkv: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    scaling,                     # float32
    BLOCK_N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # one program per (b, h, m)
    total = B * H
    b = pid // H
    h = pid % H
    m = tl.program_id(axis=2)  # axis2 is S
    # Load q for this m
    q = tl.load(Q_ptr + b * H * S * D + h * S * D + m * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
    # For each n tile, compute scores for each n, softmax per element, store vector
    for n_start in range(0, S, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        # Map attention head h to KV head kv_h: group = h // NUM_GROUPS, kv_h = group * (Hkv // NUM_GROUPS)
        kv_h = (h // NUM_GROUPS) * (Hkv // NUM_GROUPS)
        # Load K[n_offs, :]
        k = tl.load(K_ptr + b * Hkv * S * D + kv_h * S * D + n_offs * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
        # Compute scores vector for this tile: q · k per element
        scores = tl.sum(q[:, None] * k[None, :], axis=1)  # [BLOCK_N]
        scores = scores * scaling
        # Apply causal mask per element: if n > m, set to -inf
        causal_mask = n_offs > m
        scores = tl.where(causal_mask, -1e20, scores)
        # Softmax per element for this tile
        max_score = tl.max(scores, axis=0)
        scores = scores - max_score
        exp_scores = tl.exp(scores)
        sum_exp = tl.sum(exp_scores, axis=0)
        softmax = exp_scores / sum_exp  # vector [BLOCK_N]
        # Store into Soft[b, h, m, n_offs] contiguous: linear offset = m*S*S + n_offs
        tl.store(Soft_ptr + m * S * S + n_offs, softmax, mask=(n_offs < S))

# Triton kernel: compute final output by reading Soft and V, output Y_out[B, H, S, D]
# Y_out_ptr should be [B, H, S, D] contiguous. We iterate over m tiles, load Soft vectors for m across n, then multiply with V tiles.
@triton.jit
def compute_output_from_softmax_and_v(
    Soft_ptr,     # *fp32, [B, H, S, S] contiguous
    V_ptr,        # *fp32, [B, H, S, D] (note: H here refers to num_key_value_heads, but we map each attention head to KV head via NUM_GROUPS)
    Y_ptr,        # *fp32, [B, H, S, D] output
    B: tl.constexpr,
    H: tl.constexpr,            # attention heads
    Hkv: tl.constexpr,          # key/value heads
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # one program per (b, h, m_tile)
    total = B * H
    b = pid // H
    h = pid % H
    m_start = tl.program_id(axis=1) * BLOCK_M
    # Loop over m in tile
    for m_off in range(0, BLOCK_M):
        m = m_start + m_off
        if m >= S:
            break
        # For each m, we need Soft[b, h, m, :] across all n. We'll loop over n tiles and accumulate.
        out = tl.zeros([D], dtype=tl.float32)
        for n_start in range(0, S, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            # Load Soft[b, h, m, n_offs] vector [BLOCK_N]
            softmax_vec = tl.load(Soft_ptr + m * S * S + n_offs, mask=(n_offs < S), other=0.0)  # [BLOCK_N]
            # Map attention head h to KV head kv_h: group = h // NUM_GROUPS, kv_h = group * (Hkv // NUM_GROUPS)
            kv_h = (h // NUM_GROUPS) * (Hkv // NUM_GROUPS)
            # Load V[m, n_offs, :] block: V[b, kv_h, m, n_offs] but attention head h corresponds to KV head kv_h
            # Value head is kv_h because we repeated KV heads earlier. The original code repeats KV per attention head; however, attention uses value from the repeated KV head. In GQA, each query head uses keys/values from its corresponding KV head. The original code expands to match H, but here we map each attention head h to a KV head via NUM_GROUPS and use that V.
            v = tl.load(V_ptr + b * Hkv * S * D + kv_h * S * D + m * D + n_offs * D + tl.arange(0, D), mask=(n_offs < S) & (tl.arange(0, D) < D), other=0.0)  # [BLOCK_N, D]
            # out += sum over n of softmax[m, n] * V[m, n, :]
            # We need to multiply softmax_vec[n] with v[n, :] and accumulate
            # softmax_vec: [BLOCK_N], v: [BLOCK_N, D]
            # Multiply elementwise per n: [BLOCK_N, D]
            # Triton supports broadcasting: softmax_vec[:, None] * v
            contrib = softmax_vec[:, None] * v
            # Reduce over n dimension: sum along axis=0 to get [D]
            out += tl.sum(contrib, axis=0)
        # Store Y[b, h, m, :]
        tl.store(Y_ptr + b * H * S * D + h * S * D + m * D + tl.arange(0, D), out, mask=tl.arange(0, D) < D)

# Triton kernel: output projection (no bias) Y = X @ W^T
@triton.jit
def linear_no_bias_kernel(
    X_ptr,        # *fp32, input [B, S, D_in] flattened
    W_ptr,        # *fp32, weight [D_out, D_in]
    Y_ptr,        # *fp32, output [B, S, D_out] flattened
    Bsz: tl.constexpr,
    S: tl.constexpr,
    D_in: tl.constexpr,
    D_out: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for d_out_start in range(0, D_out, BLOCK_N):
        d_out_offsets = d_out_start + tl.arange(0, BLOCK_N)
        for d_in_start in range(0, D_in, 64):
            d_in_offsets = d_in_start + tl.arange(0, 64)
            x = tl.load(X_ptr + b * S * D_in + s * D_in + d_in_offsets, mask=d_in_offsets < D_in, other=0.0)  # [64]
            w = tl.load(W_ptr + d_out_offsets[:, None] * D_in + d_in_offsets[None, :], mask=(d_out_offsets[:, None] < D_out) & (d_in_offsets[None, :] < D_in), other=0.0)
            acc += tl.sum(x[:, None] * w, axis=1)
        # Store Y[b, s, d_out_offsets]
        tl.store(Y_ptr + b * S * D_out + s * D_out + d_out_offsets, acc, mask=d_out_offsets < D_out)

# ModelNew: Triton-optimized version
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_attention_heads = NUM_ATTENTION_HEADS
        self.num_key_value_heads = NUM_KEY_VALUE_HEADS
        self.head_dim = HEAD_DIM
        self.num_key_value_groups = NUM_KEY_VALUE_GROUPS

    def forward(self, hidden_states, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias, v_proj_weight, v_proj_bias, o_proj_weight, q_norm_weight, k_norm_weight, cos, sin, rms_norm_eps):
        B, S, D_in = hidden_states.shape
        assert D_in == self.num_attention_heads * self.head_dim, "hidden_states last dim must be num_attention_heads * head_dim"
        assert self.num_attention_heads % self.num_key_value_groups == 0, "num_attention_heads must be divisible by num_key_value_groups"
        # Dense projections in Triton
        # Output dims: [B, S, D_out] where D_out = head_dim
        # Q = hidden_states @ q_proj_weight^T + q_proj_bias
        Q = torch.empty((B, S, self.head_dim), device=hidden_states.device, dtype=DTYPE)
        grid_q = (B * S,)
        linear_kernel[grid_q](hidden_states, q_proj_weight, q_proj_bias if q_proj_bias is not None else torch.empty(0, device=hidden_states.device, dtype=DTYPE), Q, B, S, D_in, self.head_dim, 64, 64)

        # K = hidden_states @ k_proj_weight^T + k_proj_bias
        K = torch.empty((B, S, self.head_dim), device=hidden_states.device, dtype=DTYPE)


def run(*args):
    return ModelNew()(*args)
