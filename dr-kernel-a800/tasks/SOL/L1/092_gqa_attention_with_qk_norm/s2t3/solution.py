import torch
import triton
import triton.language as tl

# GEMM Triton kernel: Y[M, N] = X[M, K] @ W[N, K]^T (no bias)
@triton.jit
def linear_gemm_kernel(
    x_ptr,      # *f32, shape [M, K]
    w_ptr,      # *f32, shape [N, K]
    y_ptr,      # *f32, shape [M, N]
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        w_ptrs = w_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < K), other=0.0)
        acc += tl.dot(x, w)

    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# RMSNorm per-row (last dim) with per-head weight and eps: y[M, N] = (x[M, N] / sqrt(mean(x^2)+eps)) * w[M, N]
@triton.jit
def rmsnorm_heads_kernel(
    x_ptr,      # *f32, shape [M, N]
    w_ptr,      # *f32, shape [M, N] (but intended as per-head per-row; we pass per-head vectors broadcast over rows)
    y_ptr,      # *f32, shape [M, N]
    M, N, eps,
    stride_xm, stride_xn,
    stride_wm, stride_wn,
    stride_ym, stride_yn,
    per_row_weight: tl.constexpr,  # 0=query, 1=key
):
    # For per-head weight, we pass the correct weight tensor via w_ptr (e.g., q_norm_weight or k_norm_weight).
    # We also pass per_row_weight to select between query and key weights (meta-arg).
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
    w_ptrs = w_ptr + (offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn)
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    w = tl.load(w_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=1.0)

    # Compute per-row RMS
    x2 = x * x
    mean_x2 = tl.sum(x2, axis=1)  # reduce over N
    rms = tl.sqrt(mean_x2 + eps)  # shape [BLOCK_M]
    inv_rms = 1.0 / rms[:, None]  # broadcast over N

    y = x * inv_rms * w
    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, y, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Rotate half on last dimension D=128: y = [cos, sin] * [q1, -q2; q2, q1]
# We pass x_ptr (query or key), y_ptr, cos_ptr, sin_ptr (length D), D.
@triton.jit
def rotate_half_kernel(
    x_ptr,      # *f32, shape [M, D], but here M=1 since we rotate a single row-vector per (b,h)
    y_ptr,      # *f32, shape [M, D]
    cos_ptr,    # *f32, length D
    sin_ptr,    # *f32, length D
    D,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # We implement row-wise rotation. Grid dimension 0 is rows, dimension 1 over columns in tiles.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Load x row
    x_ptrs = x_ptr + pid_m * stride_xm + offs_n * stride_xn
    x = tl.load(x_ptrs, mask=offs_n < D, other=0.0)

    # Split into two halves
    half = D // 2
    q1 = x[:half]
    q2 = x[half:]
    # Load cos and sin halves
    cos1 = tl.load(cos_ptr + tl.arange(0, half))
    cos2 = tl.load(cos_ptr + tl.arange(0, half) + half)
    sin1 = tl.load(sin_ptr + tl.arange(0, half))
    sin2 = tl.load(sin_ptr + tl.arange(0, half) + half)

    # Rotated halves: q1' = cos1*q1 - sin1*q2, q2' = cos2*q2 + sin2*q1
    q1_rot = cos1 * q1 - sin1 * q2
    q2_rot = cos2 * q2 + sin2 * q1

    y = tl.zeros((D,), dtype=tl.float32)
    y[:half] = q1_rot
    y[half:] = q2_rot

    y_ptrs = y_ptr + pid_m * stride_ym + offs_n * stride_yn
    tl.store(y_ptrs, y, mask=offs_n < D)


# Attention matmul for a single position s: out[M, N] = q[M, K] @ k[M, K]^T, where M=B*S, K=head_dim (128)
@triton.jit
def attn_matmul_s_kernel(
    q_ptr,      # *f32, shape [M, K]
    k_ptr,      # *f32, shape [M, K]
    out_ptr,    # *f32, shape [M, N]
    M, N, K,
    scale,      # float32
    stride_qm, stride_qk,
    stride_km, stride_kk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        q_ptrs = q_ptr + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        k_ptrs = k_ptr + (offs_n[None, :] * stride_km + offs_k[:, None] * stride_kk)
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        k = tl.load(k_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < K), other=0.0)
        acc += tl.dot(q, k)

    acc = acc * scale
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Row-wise softmax over last dimension N for each of M rows: y = softmax(x / inv_temp, dim=-1)
@triton.jit
def softmax_row_kernel(
    x_ptr,      # *f32, shape [M, N]
    y_ptr,      # *f32, shape [M, N]
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    inv_temp,   # float32
):
    pid_m = tl.program_id(0)  # we launch grid (M, 1) so pid_m indexes rows
    # Compute max for numerical stability
    row_max = -float('inf')
    for n in range(0, N):
        val = tl.load(x_ptr + pid_m * stride_xm + n * stride_xn)
        row_max = tl.maximum(row_max, val)
    # Compute exp and sum
    sum_exp = 0.0
    for n in range(0, N):
        val = tl.load(x_ptr + pid_m * stride_xm + n * stride_xn)
        exp_val = tl.exp((val - row_max) * inv_temp)
        sum_exp += exp_val
    # Write normalized
    for n in range(0, N):
        val = tl.load(x_ptr + pid_m * stride_xm + n * stride_xn)
        y_val = tl.exp((val - row_max) * inv_temp) / sum_exp
        tl.store(y_ptr + pid_m * stride_ym + n * stride_yn, y_val)


# Output projection: y[M, N] = x[M, K] @ w[N, K]^T, no bias
@triton.jit
def output_proj_kernel(
    x_ptr,      # *f32, shape [M, K]
    w_ptr,      # *f32, shape [N, K]
    y_ptr,      # *f32, shape [M, N]
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        w_ptrs = w_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < K), other=0.0)
        acc += tl.dot(x, w)

    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def __init__(self, num_attention_heads=96, head_dim=128, num_key_value_heads=8, num_key_value_groups=12):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_proj_weight: torch.Tensor,
        q_proj_bias: torch.Tensor,
        k_proj_weight: torch.Tensor,
        k_proj_bias: torch.Tensor,
        v_proj_weight: torch.Tensor,
        v_proj_bias: torch.Tensor,
        o_proj_weight: torch.Tensor,
        q_norm_weight: torch.Tensor,   # [num_attention_heads, head_dim] = [96, 128]
        k_norm_weight: torch.Tensor,   # [num_key_value_heads, head_dim] = [8, 128]
        cos: torch.Tensor,             # [head_dim] = [128]
        sin: torch.Tensor,             # [head_dim] = [128]
        rms_norm_eps: float,
    ):
        # Shapes
        B, S, H = hidden_states.shape
        assert H == self.num_attention_heads * self.head_dim, "hidden_dim must equal num_attention_heads * head_dim"
        D = self.head_dim

        # 1) Linear projections: query, key, value (no bias), use Triton GEMM
        # Flatten [B, S, H] -> [M, K] where M = B*S, K = H
        M = B * S
        K = H
        hidden_flat = hidden_states.reshape(M, K)

        # Output tensors for query, key, value
        query = torch.empty_like(hidden_flat)
        key = torch.empty_like(hidden_flat)
        value = torch.empty_like(hidden_flat)

        # Launch GEMM kernels
        # Grid: (ceil_div(M, BLOCK_M), ceil_div(K, BLOCK_N))
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))

        # query
        linear_gemm_kernel[grid](
            hidden_flat, q_proj_weight, query, M, K, K,
            hidden_flat.stride(0), hidden_flat.stride(1),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            query.stride(0), query.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # key
        linear_gemm_kernel[grid](
            hidden_flat, k_proj_weight, key, M, K, K,
            hidden_flat.stride(0), hidden_flat.stride(1),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            key.stride(0), key.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # value
        linear_gemm_kernel[grid](
            hidden_flat, v_proj_weight, value, M, K, K,
            hidden_flat.stride(0), hidden_flat.stride(1),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            value.stride(0), value.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Reshape back to [B, S, H]
        query = query.reshape(B, S, H)
        key = key.reshape(B, S, H)
        value = value.reshape(B, S, H)

        # 2) RMSNorm per head on query and key (no bias). We need to normalize per [B, S, D] slice by head.
        # We implement a Triton kernel that operates over rows (last dim). We pass per-head weights for query and key.

        # Prepare output tensors for normalized query and key
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        # For query: heads = num_attention_heads = 96
        # Launch grid over (B*S, 1) for rows; we use a 2D grid (rows, heads_tiles) but here heads_tiles=1 is fine
        # However, Triton kernels typically need BLOCK_N over columns. RMSNorm is over last dim D=128, so we can use BLOCK_N=128.
        # We pass q_norm_weight with shape [96, 128]. We will launch per (b, s) row and per head h.
        # To do this, we write a small wrapper loop in Python over heads, launching a kernel for each head with correct weights.
        # We will do the same for key with k_norm_weight [8, 128].

        # For query normalization: loop over h in [0..95]
        for h in range(self.num_attention_heads):
            # Select weight: q_norm_weight[h, :] of shape [128]
            w_q = q_norm_weight[h]  # [128]
            # x is query[b, s, :], flatten M = B*S rows
            # We need to compute y[b, s, :] = x / sqrt(mean(x^2)+eps) * w_q
            # We'll launch a grid over (rows, 1) and handle entire D in one tile.
            BLOCK_M = 128  # rows
            BLOCK_N = 128  # columns

            # x_ptr: query.view(-1, D): we need per-row pointer. Easiest: write a small per-(b,s) program.
            # But Triton grid needs fixed size. We'll instead iterate in Python and launch per (b,s) row with h fixed.
            # This ensures we cover all rows. For simplicity, we vectorize over rows in tiles.

            # We'll flatten and process in chunks. But since M=B*S can be large, we better use a 2D grid across rows.
            # Implement by making a temporary 2D view: reshape query to [M, D], and launch kernel over (ceil(M/BLOCK_M), 1).
            # We need to pass weights per head. Pass w_q as a tensor pointer.

            x_ptr = query.view(M, D)
            y_ptr = query_norm.view(M, D)

            # Launch rmsnorm_heads_kernel for this head and weights
            linear_rms = rmsnorm_heads_kernel[(triton.cdiv(M, BLOCK_M), 1)](
                x_ptr, w_q, y_ptr, M, D, rms_norm_eps,
                x_ptr.stride(0), x_ptr.stride(1),
                w_q.stride(0), w_q.stride(1),
                y_ptr.stride(0), y_ptr.stride(1),
                per_row_weight=0,  # meta arg for potential future branching (not needed here)
                num_warps=4, num_stages=2
            )

        # For key normalization: loop over k_heads in [0..7]
        for k in range(self.num_key_value_heads):
            w_k = k_norm_weight[k]  # [128]
            x_ptr = key.view(M, D)
            y_ptr = key_norm.view(M, D)
            rmsnorm_heads_kernel[(triton.cdiv(M, BLOCK_M), 1)](
                x_ptr, w_k, y_ptr, M, D, rms_norm_eps,
                x_ptr.stride(0), x_ptr.stride(1),
                w_k.stride(0), w_k.stride(1),
                y_ptr.stride(0), y_ptr.stride(1),
                per_row_weight=1,
                num_warps=4, num_stages=2
            )

        # 3) Apply RoPE on query and key (last dim D=128), split into halves and rotate. Use Triton rotate_half_kernel.
        # We need to apply rotation per (b, h) for each sequence position s. Since we have [B, S, D], we can apply per row.
        # But we want per head rotation. For simplicity, we loop over heads and apply rotation on entire tensor by indexing rows.

        # Prepare rotated tensors
        query_rot = torch.empty_like(query)
        key_rot = torch.empty_like(key)

        # Loop over heads to apply rotation
        for h in range(self.num_attention_heads):
            # Rotate query
            # We need to pick rows corresponding to this head. For Q, each token has its own row; we rotate each row by head h.
            # We'll rotate query_norm: each row is [D]. Launch grid over rows M in tiles.
            BLOCK_M = 128
            BLOCK_N = 128
            grid = (triton.cdiv(M, BLOCK_M), 1)
            rotate_half_kernel[grid](
                query_norm.view(M, D), query_rot.view(M, D), cos, sin, D,
                query_norm.view(M, D).stride(0), query_norm.view(M, D).stride(1),
                query_rot.view(M, D).stride(0), query_rot.view(M, D).stride(1),
                num_warps=4, num_stages=2
            )

        for k in range(self.num_key_value_heads):
            # Rotate key similarly
            rotate_half_kernel[grid](
                key_norm.view(M, D), key_rot.view(M, D), cos, sin, D,
                key_norm.view(M, D).stride(0), key_norm.view(M, D).stride(1),
                key_rot.view(M, D).stride(0), key_rot.view(M, D).stride(1),
                num_warps=4, num_stages=2
            )

        # 4) Group Query Attention: expand key/value from 8 heads to 96 heads via replication (num_key_value_groups=12).
        # We need to repeat each of the 8 key/value heads 12 times to fill 96 heads.
        # Note: original code repeats along the head dimension (8 -> 96) by using groups. Since we don't have group info per token,
        # we will simply replicate by taking key_rot and value across 8 heads and expanding to 96 heads. The original uses num_key_value_groups
        # to map positions, but in grouped replication, the key/value content is reused across groups; here, the attention weight is
        # computed per head independently, so replicating content is acceptable for this Triton-focused demonstration.
        # We'll create expanded tensors for key_rot and value.
        # key_rot expanded: [B, S, 8, D] -> [B, 96, S, D]
        # value expanded similarly.
        # However, we do not have group assignment. In GQA, attention is computed per head; the reused key/value blocks are
        # considered by all attention heads (i.e., the content is reused). Since we normalized per head and rotated, we can simply
        # replicate the 8 key/value sets across 96 heads.

        # Create expanded key/value by repetition (simple replication without group mapping):
        # We'll form a 4D view for attention: [B, H, S, D] where H=96. Since original key/value are [B, S, 8, D], we can broadcast
        # and repeat. We'll create a list of 8 sets and repeat each 12 times (96 total), but to avoid dynamic loops in Triton, we
        # will construct these in PyTorch using expand. This is fine because it's data movement, not computation, and we can still
        # compute attention per (b,h) in Triton.

        # Expand key_rot to [B, 96, S, D]
        key_rot_expanded = key_rot.unsqueeze(2).expand(B, 8, S, D).contiguous().view(B, 8 * S, D)
        # To form [B, 96, S, D], we need to reshape: since S is not multiplied, we need to repeat along head dimension. We'll do
        # a simple trick: key_rot has shape [B, S, 8, D]; we'll expand to [B, 96, S, D] by repeating each of 8 heads 12 times.
        # However, Triton kernels operate on tensors we pass; constructing expanded tensors here is okay for demonstration.

        # We need key/value for attention: shape [B, H, S, D] with H=96. Since original uses 8 heads, we replicate content:
        # Build a tensor by repeating each of the 8 sets 12 times along head dimension. We can do this via torch.repeat_interleave,
        # but to keep Triton usage, we'll manually construct by iterating. Given H=96 and 8 heads, we can assign each original head
        # to 12 positions in the expanded list. For simplicity, we'll create a list and then convert to tensor using repeat_interleave.

        # Prepare expanded key/value as [B, 96, S, D]
        # Construct indices for repeat_interleave along head dimension
        # We want to map 8 -> 96, with each original index repeated 12 times. We can use torch.repeat_interleave to build a mapping.
        # But since we cannot rely on torch repeat_interleave here, we'll use a simple Python loop to assign.

        # Manual construction: create a tensor of zeros and fill blocks
        key_rot_expanded = torch.empty((B, 96, S, D), dtype=query_rot.dtype, device=query_rot.device)
        # Fill: for each original head k in 0..7, write into expanded heads indices [k*12 : (k+1)*12]
        for k in range(self.num_key_value_heads):
            start = k * self.num_key_value_groups
            end = start + self.num_key_value_groups
            # Since num_key_value_groups=12, end-start=12
            # We'll set key_rot_expanded[:, start:end, :, :] = key_rot[:, k, :, :]
            # But we need to assign across batch and sequence. It's easier to assign per (b, s, h') block.

        # To do this without torch operations, we will compute per (b, s) and per (k, group) assignment in Python loops. Since Triton
        # kernels must be launched, we will compute this in Python (data movement), then run Triton kernels per (b, h) to compute
        # attention.

        # 5) Compute attention for each (b, h): attention output [B, 96, S, D]
        # We'll implement attention per (b, h) across all S positions. For each s, compute attn_weights[B, 96, 1, 1] =
        # query_rot[b, h, s, :] @ key_rot_expanded[b, h, :, :]^T (shape [1, D] @ [S, D]^T -> [S, 1]), scale by 1/sqrt(D),
        # apply causal mask: for j > i, set attn_weights[b, h, j, s] = -inf. Then softmax over S axis for that vector, and
        # compute attn_output[b, h, s, :] =


def run(*args):
    return ModelNew()(*args)
