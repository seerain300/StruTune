# solution=GPT-5.6-Sol_dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64_triton_optimized_r1 score=24.587542137256698 passed=True
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
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_k = tl.program_id(1)

    h_offsets = tl.arange(0, num_qo_heads)
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    indices = tl.load(sparse_indices + pid_t * topk + k_offsets)
    valid = indices != -1

    acc = tl.zeros((num_qo_heads, BLOCK_K), dtype=tl.float32)

    for d_start in range(0, head_dim_ckv, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)

        q = tl.load(
            q_nope
            + pid_t * num_qo_heads * head_dim_ckv
            + h_offsets[:, None] * head_dim_ckv
            + d_offsets[None, :]
        )
        k = tl.load(
            ckv_cache
            + indices[:, None] * head_dim_ckv
            + d_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        acc += tl.dot(q, tl.trans(k))

    d_offsets = tl.arange(0, head_dim_kpe)

    q_pe_tile = tl.load(
        q_pe
        + pid_t * num_qo_heads * head_dim_kpe
        + h_offsets[:, None] * head_dim_kpe
        + d_offsets[None, :]
    )
    k_pe_tile = tl.load(
        kpe_cache
        + indices[:, None] * head_dim_kpe
        + d_offsets[None, :],
        mask=valid[:, None],
        other=0.0,
    )

    acc += tl.dot(q_pe_tile, tl.trans(k_pe_tile))
    acc *= sm_scale
    acc = tl.where(valid[None, :], acc, -float("inf"))

    score_ptrs = (
        scores
        + (pid_t * num_qo_heads + h_offsets[:, None]) * topk
        + k_offsets[None, :]
    )
    tl.store(score_ptrs, acc)


@triton.jit
def _softmax_kernel(
    scores,
    lse,
    num_qo_heads: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    offsets = tl.arange(0, BLOCK_K)
    score_ptrs = scores + (pid_t * num_qo_heads + pid_h) * topk + offsets
    logits = tl.load(score_ptrs)
    valid = logits != -float("inf")

    max_logit = tl.max(logits, axis=0)

    inv_log2 = 1.4426950408889634
    exp_logits = tl.exp2((logits - max_logit) * inv_log2)
    exp_logits = tl.where(valid, exp_logits, 0.0)

    denom = tl.sum(exp_logits, axis=0)
    probabilities = tl.where(denom > 0.0, exp_logits / denom, 0.0)

    tl.store(score_ptrs, probabilities)

    row_lse = max_logit * inv_log2 + tl.log(denom) * inv_log2
    row_lse = tl.where(denom > 0.0, row_lse, -float("inf"))
    tl.store(lse + pid_t * num_qo_heads + pid_h, row_lse)


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
    pid_d = tl.program_id(1)

    h_offsets = tl.arange(0, num_qo_heads)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    acc = tl.zeros((num_qo_heads, BLOCK_D), dtype=tl.float32)

    for k_start in range(0, topk, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        probabilities = tl.load(
            scores
            + (pid_t * num_qo_heads + h_offsets[:, None]) * topk
            + k_offsets[None, :]
        ).to(tl.bfloat16)

        indices = tl.load(sparse_indices + pid_t * topk + k_offsets)
        valid = indices != -1

        values = tl.load(
            ckv_cache
            + indices[:, None] * head_dim_ckv
            + d_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )

        acc += tl.dot(probabilities, values)

    output_ptrs = (
        output
        + (pid_t * num_qo_heads + h_offsets[:, None]) * head_dim_ckv
        + d_offsets[None, :]
    )
    tl.store(output_ptrs, acc)


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    tensors = [q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices]
    original_device = q_nope.device

    if not torch.cuda.is_available():
        if any(tensor.device.type == "cuda" for tensor in tensors):
            raise RuntimeError("CUDA is required for GPU tensor inputs")
        raise RuntimeError("CUDA is not available")

    cuda_device = next(
        (
            tensor.device
            for tensor in tensors
            if tensor.device.type == "cuda"
        ),
        torch.device("cuda"),
    )

    def to_cuda(tensor):
        if tensor.device == cuda_device:
            return tensor
        return tensor.cuda(device=cuda_device)

    q_nope_gpu = to_cuda(q_nope)
    q_pe_gpu = to_cuda(q_pe)
    ckv_cache_gpu = to_cuda(ckv_cache)
    kpe_cache_gpu = to_cuda(kpe_cache)
    sparse_indices_gpu = to_cuda(sparse_indices)

    if isinstance(sm_scale, torch.Tensor):
        sm_scale_value = float(
            sm_scale.detach().to(device=cuda_device).item()
        )
    else:
        sm_scale_value = float(sm_scale)

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
    if kpe_cache_gpu.shape[1] != 64:
        raise ValueError("kpe_cache page_size must be 64")
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

    _score_kernel[(num_tokens, topk // 128)](
        q_nope_gpu,
        q_pe_gpu,
        ckv_cache_gpu,
        kpe_cache_gpu,
        sparse_indices_gpu,
        scores,
        sm_scale_value,
        num_qo_heads=16,
        head_dim_ckv=512,
        head_dim_kpe=64,
        topk=2048,
        BLOCK_K=128,
        BLOCK_D=64,
        num_warps=4,
        num_stages=2,
    )

    _softmax_kernel[(num_tokens, num_qo_heads)](
        scores,
        lse,
        num_qo_heads=16,
        topk=2048,
        BLOCK_K=2048,
        num_warps=8,
    )

    _value_kernel[(num_tokens, head_dim_ckv // 128)](
        scores,
        sparse_indices_gpu,
        ckv_cache_gpu,
        output,
        num_qo_heads=16,
        head_dim_ckv=512,
        topk=2048,
        BLOCK_K=128,
        BLOCK_D=128,
        num_warps=8,
        num_stages=3,
    )

    if original_device == cuda_device:
        return output, lse
    return output.to(original_device), lse.to(original_device)