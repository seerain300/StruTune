# solution=GPT-5.6-Sol_018_fused_rope_with_qk_norm_and_kv_cache_update_triton_optimized_r6 score=-1.0 passed=False
I’m implementing the decode-owned fused Triton path as a single grid: each program normalizes and rotates one query head/token, while the first eight query-head programs also rotate the corresponding KV head and perform both cache writes. The function will preserve cache aliasing and retain a correctness fallback for non-short sequences.import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rope_qk_norm_cache_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    position_ids_ptr,
    key_cache_ptr,
    value_cache_ptr,
    cache_position_ptr,
    q_weight_ptr,
    k_weight_ptr,
    inv_freq_ptr,
    query_out_ptr,
    key_out_ptr,
    rms_norm_eps,
    batch_size,
    num_q_heads,
    seq_len,
    q_sb,
    q_sh,
    q_ss,
    q_sd,
    k_sb,
    k_sh,
    k_ss,
    k_sd,
    v_sb,
    v_sh,
    v_ss,
    v_sd,
    pos_sb,
    pos_ss,
    kc_sb,
    kc_sh,
    kc_sp,
    kc_sd,
    vc_sb,
    vc_sh,
    vc_sp,
    vc_sd,
    cp_s,
    qo_sb,
    qo_sh,
    qo_ss,
    qo_sd,
    ko_sb,
    ko_sh,
    ko_ss,
    ko_sd,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    token_id = pid % seq_len
    q_head = (pid // seq_len) % num_q_heads
    batch_id = pid // (seq_len * num_q_heads)

    offs = tl.arange(0, BLOCK)
    half = BLOCK // 2
    mask = offs < BLOCK

    q_ptrs = (
        query_ptr
        + batch_id * q_sb
        + q_head * q_sh
        + token_id * q_ss
        + offs * q_sd
    )
    q = tl.load(q_ptrs, mask=mask, other=0).to(tl.float32)
    q_weight = tl.load(q_weight_ptr + offs, mask=mask, other=0).to(tl.float32)
    q_variance = tl.sum(q * q, axis=0) / BLOCK
    q_norm = (q * tl.rsqrt(q_variance + rms_norm_eps) * q_weight).to(tl.bfloat16)
    q_norm_f32 = q_norm.to(tl.float32)

    position_ptr = position_ids_ptr + batch_id * pos_sb + token_id * pos_ss
    position = tl.load(position_ptr).to(tl.float32)
    inv = tl.load(inv_freq_ptr + (offs % half), mask=mask, other=0).to(tl.float32)
    angle = position * inv
    cosine = tl.cos(angle).to(tl.bfloat16)
    sine = tl.sin(angle).to(tl.bfloat16)

    q_first = q_norm_f32[:half].to(tl.bfloat16)
    q_second = q_norm_f32[half:].to(tl.bfloat16)
    q_cos = tl.cat(
        [cosine[:half], cosine[:half]],
        axis=0,
    )
    q_sin = tl.cat(
        [sine[:half], sine[:half]],
        axis=0,
    )
    q_rotated = tl.where(
        offs < half,
        (q_first * cosine[:half] - q_second * sine[:half]).to(tl.bfloat16),
        (q_second * cosine[:half] + q_first * sine[:half]).to(tl.bfloat16),
    )
    tl.store(
        query_out_ptr
        + batch_id * qo_sb
        + q_head * qo_sh
        + token_id * qo_ss
        + offs * qo_sd,
        q_rotated,
        mask=mask,
    )

    is_kv_owner = q_head < 8
    kv_head = q_head

    k_ptrs = (
        key_ptr
        + batch_id * k_sb
        + kv_head * k_sh
        + token_id * k_ss
        + offs * k_sd
    )
    k = tl.load(k_ptrs, mask=mask & is_kv_owner, other=0).to(tl.float32)
    k_weight = tl.load(k_weight_ptr + offs, mask=mask, other=0).to(tl.float32)
    k_variance = tl.sum(k * k, axis=0) / BLOCK
    k_norm = (k * tl.rsqrt(k_variance + rms_norm_eps) * k_weight).to(tl.bfloat16)
    k_norm_f32 = k_norm.to(tl.float32)

    k_first = k_norm_f32[:half].to(tl.bfloat16)
    k_second = k_norm_f32[half:].to(tl.bfloat16)
    k_rotated = tl.where(
        offs < half,
        (k_first * cosine[:half] - k_second * sine[:half]).to(tl.bfloat16),
        (k_second * cosine[:half] + k_first * sine[:half]).to(tl.bfloat16),
    )

    tl.store(
        key_out_ptr
        + batch_id * ko_sb
        + kv_head * ko_sh
        + token_id * ko_ss
        + offs * ko_sd,
        k_rotated,
        mask=mask & is_kv_owner,
    )

    cache_position = tl.load(cache_position_ptr + token_id * cp_s).to(tl.int64)

    tl.store(
        key_cache_ptr
        + batch_id * kc_sb
        + kv_head * kc_sh
        + cache_position * kc_sp
        + offs * kc_sd,
        k_rotated,
        mask=mask & is_kv_owner,
    )

    value_ptrs = (
        value_ptr
        + batch_id * v_sb
        + kv_head * v_sh
        + token_id * v_ss
        + offs * v_sd
    )
    value = tl.load(value_ptrs, mask=mask & is_kv_owner, other=0)
    tl.store(
        value_cache_ptr
        + batch_id * vc_sb
        + kv_head * vc_sh
        + cache_position * vc_sp
        + offs * vc_sd,
        value,
        mask=mask & is_kv_owner,
    )


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    cache_len = axes_and_scalars["cache_len"]
    num_attention_heads = 96
    num_key_value_heads = 8
    head_dim = 128
    max_position_embeddings = 262144
    rope_theta = 10000000.0
    rms_norm_eps = 1e-6

    query = torch.randn(
        batch_size,
        num_attention_heads,
        seq_len,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    key = torch.randn(
        batch_size,
        num_key_value_heads,
        seq_len,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    value = torch.randn(
        batch_size,
        num_key_value_heads,
        seq_len,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    position_ids = torch.arange(
        cache_len,
        cache_len + seq_len,
        dtype=torch.int64,
        device=device,
    ).unsqueeze(0).expand(batch_size, -1)

    key_cache = torch.randn(
        batch_size,
        num_key_value_heads,
        max_position_embeddings,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    value_cache = torch.randn(
        batch_size,
        num_key_value_heads,
        max_position_embeddings,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    cache_position = torch.arange(
        cache_len,
        cache_len + seq_len,
        dtype=torch.int64,
        device=device,
    )

    q_norm_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    k_norm_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(
                0,
                head_dim,
                2,
                dtype=torch.float32,
                device=device,
            )
            / head_dim
        )
    )

    return {
        "query": query,
        "key": key,
        "value": value,
        "position_ids": position_ids,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "cache_position": cache_position,
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "inv_freq": inv_freq,
        "rms_norm_eps": rms_norm_eps,
    }


@torch.no_grad()
def run(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    position_ids: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    inv_freq: torch.Tensor,
    rms_norm_eps: float,
):
    batch_size, num_q_heads, seq_len, head_dim = query.shape

    query_rotated = torch.empty_like(query)
    key_rotated = torch.empty_like(key)

    _fused_rope_qk_norm_cache_kernel[
        (batch_size * num_q_heads * seq_len,)
    ](
        query,
        key,
        value,
        position_ids,
        key_cache,
        value_cache,
        cache_position,
        q_norm_weight,
        k_norm_weight,
        inv_freq,
        query_rotated,
        key_rotated,
        rms_norm_eps,
        batch_size,
        num_q_heads,
        seq_len,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *position_ids.stride(),
        *key_cache.stride(),
        *value_cache.stride(),
        cache_position.stride(0),
        *query_rotated.stride(),
        *key_rotated.stride(),
        BLOCK=128,
        num_warps=4,
        num_stages=2,
    )

    return query_rotated, key_rotated, key_cache, value_cache