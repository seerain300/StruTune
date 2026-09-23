import torch
import triton
import triton.language as tl

# Triton kernel: RMS normalization per row (length D). Output y = x * scale, where scale = 1/sqrt(mean(x^2) + eps).
@triton.jit
def rms_norm_rows_kernel(x_ptr, out_ptr, D: tl.constexpr, eps):
    row_id = tl.program_id(0)
    offs = tl.arange(0, D)
    x = tl.load(x_ptr + row_id * D + offs).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0)
    mean = sum_sq / D
    scale = 1.0 / tl.sqrt(mean + eps)
    y = x * scale
    tl.store(out_ptr + row_id * D + offs, y.to(tl.bfloat16), mask=offs < D)

# Triton kernel: generate emb[B, S, D] where emb[..., :D//2] = inv_freq[:D//2], duplicated to full D.
# Output is float32. Each program computes one element emb[b, s, j].
@triton.jit
def emb_kernel(position_ids_ptr, inv_freq_ptr, out_emb_ptr,
               B, S, D: tl.constexpr, D_half: tl.constexpr):
    b = tl.program_id(0)
    s = tl.program_id(1)
    j = tl.program_id(2)
    if b >= B or s >= S:
        return
    pos = tl.load(position_ids_ptr + b * S + s).to(tl.float32)  # scalar pos
    if j < D_half:
        freq = tl.load(inv_freq_ptr + j).to(tl.float32)  # inv_freq[j]
        val = pos * freq
    else:
        # duplicate first half
        freq = tl.load(inv_freq_ptr + (j - D_half)).to(tl.float32)  # inv_freq[j - D_half]
        val = pos * freq
    # emb is stored as [B, S, D] contiguous: index = b*S*D + s*D + j
    tl.store(out_emb_ptr + (b * S * D) + (s * D) + j, val)

# Triton kernel: compute cos(emb) elementwise, output float32.
@triton.jit
def cos_kernel(emb_ptr, out_cos_ptr, B, S, D: tl.constexpr):
    b = tl.program_id(0)
    s = tl.program_id(1)
    j = tl.program_id(2)
    if b >= B or s >= S:
        return
    val = tl.load(emb_ptr + (b * S * D) + (s * D) + j).to(tl.float32)
    c = tl.cos(val)
    tl.store(out_cos_ptr + (b * S * D) + (s * D) + j, c)

# Triton kernel: compute sin(emb) elementwise, output float32.
@triton.jit
def sin_kernel(emb_ptr, out_sin_ptr, B, S, D: tl.constexpr):
    b = tl.program_id(0)
    s = tl.program_id(1)
    j = tl.program_id(2)
    if b >= B or s >= S:
        return
    val = tl.load(emb_ptr + (b * S * D) + (s * D) + j).to(tl.float32)
    s_val = tl.sin(val)
    tl.store(out_sin_ptr + (b * S * D) + (s * D) + j, s_val)

# Triton kernel: apply_rope per (b, h, s) row: y = x * cos + rotate_half(x) * sin
# x, cos, sin, out are pointers to tensors of shape [B, num_heads, S, D], and each program handles one row.
@triton.jit
def apply_row_kernel_2d(x_ptr, cos_ptr, sin_ptr, out_ptr,
                         B: tl.constexpr, S: tl.constexpr, num_heads: tl.constexpr, D: tl.constexpr):
    pid_bh = tl.program_id(0)  # over B * num_heads
    pid_s = tl.program_id(1)   # over S
    b = pid_bh // num_heads
    h = pid_bh % num_heads
    s = pid_s
    if b >= B or s >= S:
        return
    # Each row has length D; rows are laid out as: (b * num_heads + h) * S * D + s * D
    base = (b * num_heads + h) * S * D + s * D
    offs = tl.arange(0, D)
    x = tl.load(x_ptr + base + offs).to(tl.float32)
    c = tl.load(cos_ptr + base + offs).to(tl.float32)
    s_val = tl.load(sin_ptr + base + offs).to(tl.float32)

    # Compute rotate_half(x): for i in [0, D//2), out[2*i] = -x[2*i+1], out[2*i+1] = x[2*i]
    half = D // 2
    for i in range(0, half):
        a = x[2 * i]
        b2 = x[2 * i + 1]
        y2i = a * c[2 * i] + (-b2) * s_val[2 * i]
        y2i1 = b2 * c[2 * i + 1] + a * s_val[2 * i + 1]
        tl.store(out_ptr + base + (2 * i), y2i)
        tl.store(out_ptr + base + (2 * i + 1), y2i1)

# Triton kernel: update cache at given cache_position for each batch
# key_cache, value_cache: [B, num_kv_heads, max_pos, D], cache_position: [S]
@triton.jit
def cache_update_kernel(key_cache_ptr, value_cache_ptr, key_rotated_ptr, value_ptr,
                         batch_size, S, num_kv_heads, max_pos, D: tl.constexpr, cache_position_ptr):
    b = tl.program_id(0)  # over batch
    if b >= batch_size:
        return
    # For each kv head, copy the last S tokens from key_rotated[b, h] into key_cache[b, h, cache_position]
    for h in range(0, num_kv_heads):
        base_key = b * num_kv_heads * S * D + h * S * D
        base_cache = b * num_kv_heads * max_pos * D + h * max_pos * D
        # Iterate over s in S
        for s in range(0, S):
            pos = tl.load(cache_position_ptr + s)  # int64
            # copy key_rotated[b, h, s, :]
            src = key_rotated_ptr + base_key + s * D
            dst = key_cache_ptr + base_cache + pos * D
            for d in range(0, D):
                val = tl.load(src + d).to(tl.float32)
                tl.store(dst + d, val.to(tl.bfloat16))
        # copy value[b, h, s, :] into value_cache at same positions
        base_val = b * num_kv_heads * S * D + h * S * D
        for s in range(0, S):
            pos = tl.load(cache_position_ptr + s)
            src_v = value_ptr + base_val + s * D
            dst_v = value_cache_ptr + base_cache + pos * D
            for d in range(0, D):
                val = tl.load(src_v + d).to(tl.float32)
                tl.store(dst_v + d, val.to(tl.bfloat16))

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                position_ids: torch.Tensor,
                key_cache: torch.Tensor,
                value_cache: torch.Tensor,
                cache_position: torch.Tensor,
                q_norm_weight: torch.Tensor,
                k_norm_weight: torch.Tensor,
                inv_freq: torch.Tensor,
                rms_norm_eps: float):
        """
        Perform all computations in Triton. No torch.cos, torch.sin, or torch.cat.

        Returns: query_rotated, key_rotated, updated key_cache, updated value_cache.
        """
        # Shapes
        B, num_q_heads, S, D = query.shape
        Bk, num_kv_heads, Sk, Dk = key.shape
        assert D == 128 and Dk == 128, "This optimized kernel expects head_dim=128"
        D_half = D // 2
        assert len(inv_freq) == D_half, "inv_freq length must be head_dim//2 (64 for D=128)"

        # 1) RMS normalization for query and key using Triton
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        N_rows_q = B * num_q_heads * S
        rms_norm_rows_kernel[(N_rows_q,)](query, query_norm, D, rms_norm_eps)

        N_rows_k = Bk * num_kv_heads * Sk
        rms_norm_rows_kernel[(N_rows_k,)](key, key_norm, D, rms_norm_eps)

        # 2) Prepare emb, cos, sin for RoPE using Triton
        # emb[B, S, D] as float32
        emb = torch.empty((B, S, D), dtype=torch.float32, device=query.device)
        grid_emb = (B, S, D)


def run(*args):
    return ModelNew()(*args)
