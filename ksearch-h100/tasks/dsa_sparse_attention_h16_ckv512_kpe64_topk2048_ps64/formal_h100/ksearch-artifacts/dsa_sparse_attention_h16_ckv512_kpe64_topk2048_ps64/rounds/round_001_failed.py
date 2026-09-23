# solution=GPT-5.6-Sol_dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64_triton_optimized_r1 score=-1.0 passed=False
I’m implementing the requested materialized path as three Triton stages: score construction into an FP32 buffer, per-row softmax/LSE, and weighted-value accumulation. The wrapper will normalize devices and restore outputs to the query’s original device.import math
import torch
import triton
import triton.language as tl


@triton.jit
def _score_kernel(
    q_nope,
    q_pe,
    ckv_cache,
    kpe_cache,
    sparse_indices,
    scores,
    sm_scale,
    num_qo_heads: tl.constexpr,
    head_dim_ckv: tl.constexpr,
    head_dim_kpe: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D_CKV: tl.constexpr,
    BLOCK_D_KPE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_k = tl.program_id(2)

    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < topk

    index_ptr = sparse_indices + pid_t * topk + k_offsets
    indices = tl.load(index_ptr, mask=k_mask, other=0)
    valid = k_mask & (indices != -1)

    ckv_base = indices[:, None] * head_dim_ckv
    kpe_base = indices[:, None] * head_dim_kpe

    score = tl.zeros([BLOCK_K], dtype=tl.float32)

    for d_start in range(0, head_dim_ckv, BLOCK_D_CKV):
        d_offsets = d_start + tl.arange(0, BLOCK_D_CKV)
        d_mask = d_offsets < head_dim_ckv

        q_ptr = (
            q_nope
            + pid_t * num_qo_heads * head_dim_ckv
            + pid_h * head_dim_ckv
            + d_offsets
        )
        q = tl.load(q_ptr, mask=d_mask, other=0.0).to(tl.float32)

        k_ptr = ckv_cache + ckv_base + d_offsets[None, :]
        k = tl.load(
            k_ptr,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        score += tl.sum(k * q[None, :], axis=1)

    for d_start in range(0, head_dim_kpe, BLOCK_D_KPE):
        d_offsets = d_start + tl.arange(0, BLOCK_D_KPE)
        d_mask = d_offsets < head_dim_kpe

        q_ptr = (
            q_pe
            + pid_t * num_qo_heads * head_dim_kpe
            + pid_h * head_dim_kpe
            + d_offsets
        )
        q = tl.load(q_ptr, mask=d_mask, other=0.0).to(tl.float32)

        k_ptr = kpe_cache + kpe_base + d_offsets[None, :]
        k = tl.load(
            k_ptr,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        score += tl.sum(k * q[None, :], axis=1)

    score = score * sm_scale
    score = tl.where(valid, score, -float("inf"))

    score_ptr = scores + (pid_t * num_qo_heads + pid_h) * topk + k_offsets
    tl.store(score_ptr, score, mask=k_mask)


@triton.jit
def _softmax_kernel(
    scores,
    sparse_indices,
    lse,
    num_qo_heads: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < topk

    score_ptr = scores + (pid_t * num_qo_heads + pid_h) * topk + offsets
    index_ptr = sparse_indices + pid_t * topk + offsets

    logits = tl.load(score_ptr, mask=mask, other=-float("inf"))
    indices = tl.load(index_ptr, mask=mask, other=-1)
    valid = mask & (indices != -1)
    logits = tl.where(valid, logits, -float("inf"))

    max_logit = tl.max(logits, axis=0)
    exp_logits = tl.exp2((logits - max_logit) * 1.4426950408889634)
    exp_logits = tl.where(valid, exp_logits, 0.0)

    denom = tl.sum(exp_logits, axis=0)
    probs = exp_logits / denom

    tl.store(score_ptr, probs, mask=mask)

    row_lse = (
        max_logit * 1.4426950408889634
        + tl.log(denom) * 1.4426950408889634
    )
    row_lse = tl.where(denom > 0.0, row_lse, -float("inf"))

    lse_ptr = lse + pid_t * num_qo_heads + pid_h
    tl.store(lse_ptr, row_lse)


@triton.jit
def _value_kernel(
    scores,
    sparse_indices,
    ckv_cache,
    output,
    num_qo_heads: tl.constexpr,
    head_dim_ckv: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim_ckv
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for k_start in range(0, topk, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < topk

        score_ptr = scores + (pid_t * num_qo_heads + pid_h) * topk + k_offsets
        index_ptr = sparse_indices + pid_t * topk + k_offsets

        probs = tl.load(score_ptr, mask=k_mask, other=0.0).to(tl.float32)
        indices = tl.load(index_ptr, mask=k_mask, other=0)
        valid = k_mask & (indices != -1)

        cache_ptr = ckv_cache + indices[:, None] * head_dim_ckv + d_offsets[None, :]
        values = tl.load(
            cache_ptr,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        acc += tl.sum(values * probs[:, None], axis=0)

    output_ptr = (
        output
        + pid_t * num_qo_heads * head_dim_ckv
        + pid_h * head_dim_ckv
        + d_offsets
    )
    tl.store(output_ptr, acc, mask=d_mask)


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    tensors = [q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices]
    original_device = q_nope.device

    if not torch.cuda.is_available():
        if any(t.device.type == "cuda" for t in tensors):
            raise RuntimeError("CUDA is required for GPU tensor inputs")
        raise RuntimeError("CUDA is not available")

    cuda_device = next(
        (t.device for t in tensors if t.device.type == "cuda"),
        torch.device("cuda"),
    )

    def to_cuda(tensor):
        if tensor.device.type == "cuda":
            return tensor
        return tensor.cuda(cuda_device)

    q_nope_gpu = to_cuda(q_nope)
    q_pe_gpu = to_cuda(q_pe)
    ckv_cache_gpu = to_cuda(ckv_cache)
    kpe_cache_gpu = to_cuda(kpe_cache)
    sparse_indices_gpu = to_cuda(sparse_indices)

    if isinstance(sm_scale, torch.Tensor):
        sm_scale_gpu = to_cuda(sm_scale).to(torch.float32)
    else:
        sm_scale_gpu = torch.tensor(
            float(sm_scale),
            dtype=torch.float32,
            device=cuda_device,
        )

    num_tokens, num_qo_heads, head_dim_ckv = q_nope_gpu.shape
    head_dim_kpe = q_pe_gpu.shape[-1]
    topk = sparse_indices_gpu.shape[-1]

    if num_qo_heads != 16:
        raise ValueError("num_qo_heads must be 16")
    if head_dim_ckv != 512:
        raise ValueError("head_dim_ckv must be 512")
    if head_dim_kpe != 64:
        raise ValueError("head_dim_kpe must be 64")
    if ckv_cache_gpu.shape[1] != 64:
        raise ValueError("page_size must be 64")
    if topk != 2048:
        raise ValueError("topk must be 2048")
    if sparse_indices_gpu.shape[0] != num_tokens:
        raise ValueError("sparse_indices.shape[0] must equal num_tokens")

    scores = torch.empty(
        (num_tokens, num_qo_heads, topk),
        dtype=torch.float32,
        device=cuda_device,
    )
    output = torch.empty(
        (num_tokens, num_qo_heads, head_dim_ckv),
        dtype=torch.bfloat16,
        device=cuda_device,
    )
    lse = torch.empty(
        (num_tokens, num_qo_heads),
        dtype=torch.float32,
        device=cuda_device,
    )

    score_grid = (
        num_tokens,
        num_qo_heads,
        triton.cdiv(topk, 128),
    )
    _score_kernel[score_grid](
        q_nope_gpu,
        q_pe_gpu,
        ckv_cache_gpu,
        kpe_cache_gpu,
        sparse_indices_gpu,
        scores,
        sm_scale_gpu,
        num_qo_heads=16,
        head_dim_ckv=512,
        head_dim_kpe=64,
        topk=2048,
        BLOCK_K=128,
        BLOCK_D_CKV=128,
        BLOCK_D_KPE=64,
    )

    _softmax_kernel[(num_tokens, num_qo_heads)](
        scores,
        sparse_indices_gpu,
        lse,
        num_qo_heads=16,
        topk=2048,
        BLOCK_K=2048,
    )

    value_grid = (
        num_tokens,
        num_qo_heads,
        triton.cdiv(head_dim_ckv, 128),
    )
    _value_kernel[value_grid](
        scores,
        sparse_indices_gpu,
        ckv_cache_gpu,
        output,
        num_qo_heads=16,
        head_dim_ckv=512,
        topk=2048,
        BLOCK_K=128,
        BLOCK_D=128,
    )

    return output.to(original_device), lse.to(original_device)