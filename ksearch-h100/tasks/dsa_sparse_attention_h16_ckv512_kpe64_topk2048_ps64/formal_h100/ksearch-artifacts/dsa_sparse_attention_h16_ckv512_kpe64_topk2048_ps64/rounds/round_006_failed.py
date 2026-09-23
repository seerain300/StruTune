# solution=GPT-5.6-Sol_dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64_triton_optimized_r6 score=-1.0 passed=False
I’m implementing the requested materialized decomposition as three GPU stages: indirect score generation into an FP32 buffer, a separate stable softmax/LSE pass, and a weighted-value pass. The wrapper will preserve input devices while moving execution tensors to CUDA and returning outputs on the original query device.import torch
import triton
import triton.language as tl


@triton.jit
def _score_kernel(
    q_nope_ptr,
    q_pe_ptr,
    ckv_ptr,
    kpe_ptr,
    sparse_ptr,
    scores_ptr,
    num_tokens,
    num_pages,
    sm_scale,
    HEADS: tl.constexpr,
    CKV_DIM: tl.constexpr,
    KPE_DIM: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(0)
    token_head = pid // NUM_K_BLOCKS
    k_block = pid % NUM_K_BLOCKS

    token = token_head // HEADS
    head = token_head % HEADS

    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    valid_k = k_offsets < TOPK

    sparse_offsets = token * TOPK + k_offsets
    sparse_indices = tl.load(
        sparse_ptr + sparse_offsets,
        mask=valid_k,
        other=0,
    )
    valid_indices = valid_k & (sparse_indices != -1)
    token_indices = tl.cast(sparse_indices, tl.int64)
    token_indices = tl.where(valid_indices, token_indices, 0)

    score_acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    for d0 in tl.static_range(0, CKV_DIM, BLOCK_D):
        d_offsets = d0 + tl.arange(0, BLOCK_D)
        valid_d = d_offsets < CKV_DIM

        q_ptrs = q_nope_ptr + token * HEADS * CKV_DIM + head * CKV_DIM + d_offsets
        q_values = tl.load(q_ptrs, mask=valid_d, other=0.0).to(tl.float32)

        k_ptrs = (
            ckv_ptr
            + token_indices[:, None] * CKV_DIM
            + d_offsets[None, :]
        )
        k_values = tl.load(
            k_ptrs,
            mask=valid_indices[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)

        score_acc += tl.sum(k_values * q_values[None, :], axis=1)

    for d0 in tl.static_range(0, KPE_DIM, BLOCK_D):
        d_offsets = d0 + tl.arange(0, BLOCK_D)
        valid_d = d_offsets < KPE_DIM

        q_ptrs = q_pe_ptr + token * HEADS * KPE_DIM + head * KPE_DIM + d_offsets
        q_values = tl.load(q_ptrs, mask=valid_d, other=0.0).to(tl.float32)

        k_ptrs = (
            kpe_ptr
            + token_indices[:, None] * KPE_DIM
            + d_offsets[None, :]
        )
        k_values = tl.load(
            k_ptrs,
            mask=valid_indices[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)

        score_acc += tl.sum(k_values * q_values[None, :], axis=1)

    score_acc *= sm_scale
    score_acc = tl.where(valid_indices, score_acc, -float("inf"))

    output_offsets = token_head * TOPK + k_offsets
    tl.store(scores_ptr + output_offsets, score_acc, mask=valid_k)


@triton.jit
def _softmax_kernel(
    scores_ptr,
    probs_ptr,
    lse_ptr,
    num_tokens,
    HEADS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(0)
    token = pid // HEADS
    head = pid % HEADS
    base = (token * HEADS + head) * TOPK

    max_value = -float("inf")
    for k0 in tl.static_range(0, TOPK, BLOCK_K):
        offsets = k0 + tl.arange(0, BLOCK_K)
        valid = offsets < TOPK
        values = tl.load(scores_ptr + base + offsets, mask=valid, other=-float("inf"))
        max_value = tl.maximum(max_value, tl.max(values, axis=0))

    safe_max = tl.where(max_value == -float("inf"), 0.0, max_value)
    sum_exp = 0.0

    for k0 in tl.static_range(0, TOPK, BLOCK_K):
        offsets = k0 + tl.arange(0, BLOCK_K)
        valid = offsets < TOPK
        values = tl.load(scores_ptr + base + offsets, mask=valid, other=-float("inf"))
        exp_values = tl.exp2((values - safe_max) * 1.4426950408889634)
        exp_values = tl.where(valid & (values != -float("inf")), exp_values, 0.0)
        sum_exp += tl.sum(exp_values, axis=0)

    inv_sum = tl.where(sum_exp > 0.0, 1.0 / sum_exp, 0.0)

    for k0 in tl.static_range(0, TOPK, BLOCK_K):
        offsets = k0 + tl.arange(0, BLOCK_K)
        valid = offsets < TOPK
        values = tl.load(scores_ptr + base + offsets, mask=valid, other=-float("inf"))
        exp_values = tl.exp2((values - safe_max) * 1.4426950408889634)
        exp_values = tl.where(valid & (values != -float("inf")), exp_values, 0.0)
        probabilities = exp_values * inv_sum
        tl.store(probs_ptr + base + offsets, probabilities, mask=valid)

    lse_value = tl.where(
        sum_exp > 0.0,
        safe_max * 1.4426950408889634 + tl.log2(sum_exp),
        -float("inf"),
    )
    tl.store(lse_ptr + token * HEADS + head, lse_value)


@triton.jit
def _value_kernel(
    probs_ptr,
    ckv_ptr,
    output_ptr,
    num_tokens,
    HEADS: tl.constexpr,
    CKV_DIM: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    output_block = pid % (CKV_DIM // BLOCK_D)
    token_head = pid // (CKV_DIM // BLOCK_D)

    token = token_head // HEADS
    head = token_head % HEADS

    d_offsets = output_block * BLOCK_D + tl.arange(0, BLOCK_D)
    valid_d = d_offsets < CKV_DIM
    output_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    prob_base = (token * HEADS + head) * TOPK

    for k0 in tl.static_range(0, TOPK, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        valid_k = k_offsets < TOPK

        probabilities = tl.load(
            probs_ptr + prob_base + k_offsets,
            mask=valid_k,
            other=0.0,
        ).to(tl.float32)

        sparse_indices = tl.load(
            probs_ptr - (token * HEADS + head) * TOPK + token * TOPK + k_offsets,
            mask=valid_k,
            other=0,
        )
        token_indices = tl.cast(sparse_indices, tl.int64)

        k_ptrs = (
            ckv_ptr
            + token_indices[:, None] * CKV_DIM
            + d_offsets[None, :]
        )
        k_values = tl.load(
            k_ptrs,
            mask=valid_k[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)

        output_acc += tl.sum(probabilities[:, None] * k_values, axis=0)

    output_offsets = (
        token * HEADS * CKV_DIM
        + head * CKV_DIM
        + d_offsets
    )
    tl.store(
        output_ptr + output_offsets,
        output_acc,
        mask=valid_d,
    )


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    original_output_device = q_nope.device

    if not torch.cuda.is_available():
        gpu_inputs = [
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            sparse_indices,
        ]
        if any(t.device.type == "cuda" for t in gpu_inputs):
            raise RuntimeError("CUDA is required for GPU tensor inputs.")
        raise RuntimeError("CUDA is not available; this Triton implementation requires a CUDA device.")

    if q_nope.ndim != 3:
        raise ValueError("q_nope must have shape [num_tokens, 16, 512].")
    if q_pe.ndim != 3:
        raise ValueError("q_pe must have shape [num_tokens, 16, 64].")
    if ckv_cache.ndim != 3:
        raise ValueError("ckv_cache must have shape [num_pages, 64, 512].")
    if kpe_cache.ndim != 3:
        raise ValueError("kpe_cache must have shape [num_pages, 64, 64].")
    if sparse_indices.ndim != 2:
        raise ValueError("sparse_indices must have shape [num_tokens, 2048].")

    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    q_pe_tokens, q_pe_heads, head_dim_kpe = q_pe.shape
    num_pages, page_size, ckv_dim = ckv_cache.shape
    kpe_pages, kpe_page_size, kpe_dim = kpe_cache.shape
    sparse_tokens, topk = sparse_indices.shape

    if num_qo_heads != 16 or head_dim_ckv != 512:
        raise ValueError("q_nope must have shape [num_tokens, 16, 512].")
    if (q_pe_tokens, q_pe_heads, head_dim_kpe) != (num_tokens, 16, 64):
        raise ValueError("q_pe must have shape [num_tokens, 16, 64].")
    if (page_size, ckv_dim) != (64, 512):
        raise ValueError("ckv_cache must have shape [num_pages, 64, 512].")
    if (kpe_pages, kpe_page_size, kpe_dim) != (num_pages, 64, 64):
        raise ValueError("kpe_cache must have shape [num_pages, 64, 64].")
    if (sparse_tokens, topk) != (num_tokens, 2048):
        raise ValueError("sparse_indices must have shape [num_tokens, 2048].")

    q_nope_gpu = q_nope.cuda().contiguous()
    q_pe_gpu = q_pe.cuda().contiguous()
    ckv_gpu = ckv_cache.cuda().contiguous()
    kpe_gpu = kpe_cache.cuda().contiguous()
    sparse_gpu = sparse_indices.cuda().contiguous()

    if torch.is_tensor(sm_scale):
        sm_scale_gpu = sm_scale.cuda().contiguous().to(torch.float32)
        if sm_scale_gpu.numel() != 1:
            raise ValueError("sm_scale must be a scalar tensor.")
    else:
        sm_scale_gpu = torch.tensor(
            float(sm_scale),
            dtype=torch.float32,
            device="cuda",
        )

    scores = torch.empty(
        (num_tokens, 16, 2048),
        dtype=torch.float32,
        device="cuda",
    )
    probabilities = torch.empty_like(scores)
    output_gpu = torch.empty(
        (num_tokens, 16, 512),
        dtype=torch.bfloat16,
        device="cuda",
    )
    lse_gpu = torch.empty(
        (num_tokens, 16),
        dtype=torch.float32,
        device="cuda",
    )

    block_k = 64
    block_d = 128
    num_k_blocks = 2048 // block_k

    _score_kernel[(num_tokens * 16 * num_k_blocks,)](
        q_nope_gpu,
        q_pe_gpu,
        ckv_gpu,
        kpe_gpu,
        sparse_gpu,
        scores,
        num_tokens,
        num_pages,
        sm_scale_gpu,
        HEADS=16,
        CKV_DIM=512,
        KPE_DIM=64,
        TOPK=2048,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        NUM_K_BLOCKS=num_k_blocks,
        num_warps=4,
    )

    _softmax_kernel[(num_tokens * 16,)](
        scores,
        probabilities,
        lse_gpu,
        num_tokens,
        HEADS=16,
        TOPK=2048,
        BLOCK_K=block_k,
        NUM_K_BLOCKS=num_k_blocks,
        num_warps=4,
    )

    sparse_value_ptr = sparse_gpu
    _value_kernel[(num_tokens * 16 * (512 // block_d),)](
        probabilities,
        ckv_gpu,
        output_gpu,
        num_tokens,
        HEADS=16,
        CKV_DIM=512,
        TOPK=2048,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        num_warps=4,
    )

    return (
        output_gpu.to(original_output_device),
        lse_gpu.to(original_output_device),
    )