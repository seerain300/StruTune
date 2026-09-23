# solution=GPT-5.6-Sol_gqa_paged_decode_h32_kv8_d128_ps1_triton_optimized_r3 score=537.2338164885858 passed=True
import torch
import triton
import triton.language as tl


@triton.jit
def _gqa_paged_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    output_ptr,
    lse_ptr,
    sm_scale,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_page: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_page: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_d: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    lse_stride_b: tl.constexpr,
    lse_stride_h: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    program_id = tl.program_id(0)
    batch_idx = program_id // 32
    query_head = program_id % 32
    kv_head = query_head // 4

    dims = tl.arange(0, BLOCK_D)
    q_offsets = (
        batch_idx * q_stride_b
        + query_head * q_stride_h
        + dims * q_stride_d
    )
    q = tl.load(q_ptr + q_offsets).to(tl.float32)

    page_start = tl.load(kv_indptr_ptr + batch_idx)
    page_end = tl.load(kv_indptr_ptr + batch_idx + 1)

    max_logit = -float("inf")
    denominator = 0.0
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    token_offsets = tl.arange(0, BLOCK_N)
    page_offset = page_start

    while page_offset < page_end:
        positions = page_offset + token_offsets
        token_mask = positions < page_end
        page_ids = tl.load(
            kv_indices_ptr + positions,
            mask=token_mask,
            other=0,
        )

        k_offsets = (
            page_ids[:, None] * k_stride_page
            + kv_head * k_stride_h
            + dims[None, :] * k_stride_d
        )
        k = tl.load(
            k_ptr + k_offsets,
            mask=token_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        logits = tl.sum(k * q[None, :], axis=1)
        logits = logits * sm_scale * 1.4426950408889634
        logits = tl.where(token_mask, logits, -float("inf"))

        block_max = tl.max(logits, axis=0)
        new_max = tl.maximum(max_logit, block_max)
        old_scale = tl.exp2(max_logit - new_max)
        weights = tl.exp2(logits - new_max)

        v_offsets = (
            page_ids[:, None] * v_stride_page
            + kv_head * v_stride_h
            + dims[None, :] * v_stride_d
        )
        v = tl.load(
            v_ptr + v_offsets,
            mask=token_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        accumulator = (
            accumulator * old_scale
            + tl.sum(weights[:, None] * v, axis=0)
        )
        denominator = denominator * old_scale + tl.sum(weights, axis=0)
        max_logit = new_max
        page_offset += BLOCK_N

    output_offsets = (
        batch_idx * out_stride_b
        + query_head * out_stride_h
        + dims * out_stride_d
    )
    lse_offset = batch_idx * lse_stride_b + query_head * lse_stride_h

    if page_start < page_end:
        tl.store(output_ptr + output_offsets, accumulator / denominator)
        tl.store(lse_ptr + lse_offset, max_logit + tl.log2(denominator))
    else:
        tl.store(output_ptr + output_offsets, 0.0)
        tl.store(lse_ptr + lse_offset, -float("inf"))


@triton.jit
def _gqa_paged_decode_group4_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    output_ptr,
    lse_ptr,
    sm_scale,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_page: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_page: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_d: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    lse_stride_b: tl.constexpr,
    lse_stride_h: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    program_id = tl.program_id(0)
    batch_idx = program_id // 8
    kv_head = program_id % 8
    query_head = kv_head * 4

    dims = tl.arange(0, BLOCK_D)
    q_base = (
        batch_idx * q_stride_b
        + query_head * q_stride_h
        + dims * q_stride_d
    )
    q0 = tl.load(q_ptr + q_base).to(tl.float32)
    q1 = tl.load(q_ptr + q_base + q_stride_h).to(tl.float32)
    q2 = tl.load(q_ptr + q_base + 2 * q_stride_h).to(tl.float32)
    q3 = tl.load(q_ptr + q_base + 3 * q_stride_h).to(tl.float32)

    page_start = tl.load(kv_indptr_ptr + batch_idx)
    page_end = tl.load(kv_indptr_ptr + batch_idx + 1)

    max0 = -float("inf")
    max1 = -float("inf")
    max2 = -float("inf")
    max3 = -float("inf")

    denominator0 = 0.0
    denominator1 = 0.0
    denominator2 = 0.0
    denominator3 = 0.0

    accumulator0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    accumulator1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    accumulator2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    accumulator3 = tl.zeros((BLOCK_D,), dtype=tl.float32)

    token_offsets = tl.arange(0, BLOCK_N)
    page_offset = page_start

    while page_offset < page_end:
        positions = page_offset + token_offsets
        token_mask = positions < page_end
        page_ids = tl.load(
            kv_indices_ptr + positions,
            mask=token_mask,
            other=0,
        )

        k_offsets = (
            page_ids[:, None] * k_stride_page
            + kv_head * k_stride_h
            + dims[None, :] * k_stride_d
        )
        k = tl.load(
            k_ptr + k_offsets,
            mask=token_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        logits0 = tl.sum(k * q0[None, :], axis=1)
        logits1 = tl.sum(k * q1[None, :], axis=1)
        logits2 = tl.sum(k * q2[None, :], axis=1)
        logits3 = tl.sum(k * q3[None, :], axis=1)

        logit_scale = sm_scale * 1.4426950408889634
        logits0 = tl.where(
            token_mask, logits0 * logit_scale, -float("inf")
        )
        logits1 = tl.where(
            token_mask, logits1 * logit_scale, -float("inf")
        )
        logits2 = tl.where(
            token_mask, logits2 * logit_scale, -float("inf")
        )
        logits3 = tl.where(
            token_mask, logits3 * logit_scale, -float("inf")
        )

        block_max0 = tl.max(logits0, axis=0)
        block_max1 = tl.max(logits1, axis=0)
        block_max2 = tl.max(logits2, axis=0)
        block_max3 = tl.max(logits3, axis=0)

        new_max0 = tl.maximum(max0, block_max0)
        new_max1 = tl.maximum(max1, block_max1)
        new_max2 = tl.maximum(max2, block_max2)
        new_max3 = tl.maximum(max3, block_max3)

        old_scale0 = tl.exp2(max0 - new_max0)
        old_scale1 = tl.exp2(max1 - new_max1)
        old_scale2 = tl.exp2(max2 - new_max2)
        old_scale3 = tl.exp2(max3 - new_max3)

        weights0 = tl.exp2(logits0 - new_max0)
        weights1 = tl.exp2(logits1 - new_max1)
        weights2 = tl.exp2(logits2 - new_max2)
        weights3 = tl.exp2(logits3 - new_max3)

        v_offsets = (
            page_ids[:, None] * v_stride_page
            + kv_head * v_stride_h
            + dims[None, :] * v_stride_d
        )
        v = tl.load(
            v_ptr + v_offsets,
            mask=token_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        accumulator0 = (
            accumulator0 * old_scale0
            + tl.sum(weights0[:, None] * v, axis=0)
        )
        accumulator1 = (
            accumulator1 * old_scale1
            + tl.sum(weights1[:, None] * v, axis=0)
        )
        accumulator2 = (
            accumulator2 * old_scale2
            + tl.sum(weights2[:, None] * v, axis=0)
        )
        accumulator3 = (
            accumulator3 * old_scale3
            + tl.sum(weights3[:, None] * v, axis=0)
        )

        denominator0 = (
            denominator0 * old_scale0 + tl.sum(weights0, axis=0)
        )
        denominator1 = (
            denominator1 * old_scale1 + tl.sum(weights1, axis=0)
        )
        denominator2 = (
            denominator2 * old_scale2 + tl.sum(weights2, axis=0)
        )
        denominator3 = (
            denominator3 * old_scale3 + tl.sum(weights3, axis=0)
        )

        max0 = new_max0
        max1 = new_max1
        max2 = new_max2
        max3 = new_max3
        page_offset += BLOCK_N

    output_base = (
        batch_idx * out_stride_b
        + query_head * out_stride_h
        + dims * out_stride_d
    )
    lse_base = batch_idx * lse_stride_b + query_head * lse_stride_h

    if page_start < page_end:
        tl.store(output_ptr + output_base, accumulator0 / denominator0)
        tl.store(
            output_ptr + output_base + out_stride_h,
            accumulator1 / denominator1,
        )
        tl.store(
            output_ptr + output_base + 2 * out_stride_h,
            accumulator2 / denominator2,
        )
        tl.store(
            output_ptr + output_base + 3 * out_stride_h,
            accumulator3 / denominator3,
        )

        tl.store(lse_ptr + lse_base, max0 + tl.log2(denominator0))
        tl.store(
            lse_ptr + lse_base + lse_stride_h,
            max1 + tl.log2(denominator1),
        )
        tl.store(
            lse_ptr + lse_base + 2 * lse_stride_h,
            max2 + tl.log2(denominator2),
        )
        tl.store(
            lse_ptr + lse_base + 3 * lse_stride_h,
            max3 + tl.log2(denominator3),
        )
    else:
        tl.store(output_ptr + output_base, 0.0)
        tl.store(output_ptr + output_base + out_stride_h, 0.0)
        tl.store(output_ptr + output_base + 2 * out_stride_h, 0.0)
        tl.store(output_ptr + output_base + 3 * out_stride_h, 0.0)

        tl.store(lse_ptr + lse_base, -float("inf"))
        tl.store(lse_ptr + lse_base + lse_stride_h, -float("inf"))
        tl.store(lse_ptr + lse_base + 2 * lse_stride_h, -float("inf"))
        tl.store(lse_ptr + lse_base + 3 * lse_stride_h, -float("inf"))


@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    if not isinstance(k_cache, torch.Tensor):
        raise TypeError("k_cache must be a torch.Tensor")
    if not isinstance(v_cache, torch.Tensor):
        raise TypeError("v_cache must be a torch.Tensor")
    if not isinstance(kv_indptr, torch.Tensor):
        raise TypeError("kv_indptr must be a torch.Tensor")
    if not isinstance(kv_indices, torch.Tensor):
        raise TypeError("kv_indices must be a torch.Tensor")

    tensors = (q, k_cache, v_cache, kv_indptr, kv_indices)
    if not torch.cuda.is_available():
        if any(t.is_cuda for t in tensors):
            raise RuntimeError(
                "CUDA is unavailable, but one or more inputs are CUDA tensors"
            )
        raise RuntimeError("CUDA is required to execute the Triton kernel")

    if q.ndim != 3:
        raise ValueError("q must have shape [batch_size, 32, 128]")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "k_cache and v_cache must have shape [num_pages, 1, 8, 128]"
        )

    batch_size, num_qo_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, k_head_dim = k_cache.shape

    if num_qo_heads != 32:
        raise ValueError(f"num_qo_heads must be 32, got {num_qo_heads}")
    if num_kv_heads != 8:
        raise ValueError(f"num_kv_heads must be 8, got {num_kv_heads}")
    if head_dim != 128 or k_head_dim != 128:
        raise ValueError("head_dim must be 128")
    if page_size != 1:
        raise ValueError(f"page_size must be 1, got {page_size}")
    if v_cache.shape != k_cache.shape:
        raise ValueError("v_cache must have the same shape as k_cache")
    if kv_indptr.ndim != 1 or kv_indptr.shape[0] != batch_size + 1:
        raise ValueError("len_indptr must equal batch_size + 1")
    if kv_indices.ndim != 1:
        raise ValueError("kv_indices must be one-dimensional")
    if q.dtype != torch.bfloat16:
        raise TypeError("q must have dtype torch.bfloat16")
    if k_cache.dtype != torch.bfloat16 or v_cache.dtype != torch.bfloat16:
        raise TypeError("k_cache and v_cache must have dtype torch.bfloat16")
    if kv_indptr.dtype != torch.int32 or kv_indices.dtype != torch.int32:
        raise TypeError("kv_indptr and kv_indices must have dtype torch.int32")

    original_device = q.device
    cuda_inputs = [tensor for tensor in tensors if tensor.is_cuda]
    target_device = cuda_inputs[0].device if cuda_inputs else torch.device("cuda")

    for tensor in cuda_inputs:
        if tensor.device != target_device:
            raise ValueError("all CUDA inputs must be on the same CUDA device")

    expected_indices = kv_indices.numel()
    if not kv_indptr.is_cuda:
        if int(kv_indptr[-1].item()) != expected_indices:
            raise ValueError(
                "num_kv_indices must equal kv_indptr[-1].item()"
            )
    if expected_indices and num_pages == 0:
        raise ValueError("k_cache and v_cache contain no pages")

    q_gpu = q if q.device == target_device else q.cuda(device=target_device)
    k_gpu = (
        k_cache
        if k_cache.device == target_device
        else k_cache.cuda(device=target_device)
    )
    v_gpu = (
        v_cache
        if v_cache.device == target_device
        else v_cache.cuda(device=target_device)
    )
    indptr_gpu = (
        kv_indptr
        if kv_indptr.device == target_device
        else kv_indptr.cuda(device=target_device)
    )
    indices_gpu = (
        kv_indices
        if kv_indices.device == target_device
        else kv_indices.cuda(device=target_device)
    )

    if isinstance(sm_scale, torch.Tensor):
        if sm_scale.numel() != 1:
            raise ValueError("sm_scale must be a scalar")
        sm_scale_value = float(sm_scale.item())
    else:
        sm_scale_value = float(sm_scale)

    output_gpu = torch.empty(
        (batch_size, 32, 128),
        dtype=torch.bfloat16,
        device=target_device,
    )
    lse_gpu = torch.empty(
        (batch_size, 32),
        dtype=torch.float32,
        device=target_device,
    )

    if batch_size:
        average_length = expected_indices / batch_size
        use_group4 = batch_size >= 16 and average_length >= 64

        if use_group4:
            grid = (batch_size * 8,)
            _gqa_paged_decode_group4_kernel[grid](
                q_gpu,
                k_gpu,
                v_gpu,
                indptr_gpu,
                indices_gpu,
                output_gpu,
                lse_gpu,
                sm_scale_value,
                q_gpu.stride(0),
                q_gpu.stride(1),
                q_gpu.stride(2),
                k_gpu.stride(0),
                k_gpu.stride(2),
                k_gpu.stride(3),
                v_gpu.stride(0),
                v_gpu.stride(2),
                v_gpu.stride(3),
                output_gpu.stride(0),
                output_gpu.stride(1),
                output_gpu.stride(2),
                lse_gpu.stride(0),
                lse_gpu.stride(1),
                BLOCK_D=128,
                BLOCK_N=64,
                num_warps=8,
                num_stages=1,
            )
        else:
            if average_length >= 128:
                block_n = 128
                num_warps = 8
            elif average_length >= 48:
                block_n = 64
                num_warps = 8
            elif average_length >= 32:
                block_n = 32
                num_warps = 4
            else:
                block_n = 16
                num_warps = 4

            grid = (batch_size * 32,)
            _gqa_paged_decode_kernel[grid](
                q_gpu,
                k_gpu,
                v_gpu,
                indptr_gpu,
                indices_gpu,
                output_gpu,
                lse_gpu,
                sm_scale_value,
                q_gpu.stride(0),
                q_gpu.stride(1),
                q_gpu.stride(2),
                k_gpu.stride(0),
                k_gpu.stride(2),
                k_gpu.stride(3),
                v_gpu.stride(0),
                v_gpu.stride(2),
                v_gpu.stride(3),
                output_gpu.stride(0),
                output_gpu.stride(1),
                output_gpu.stride(2),
                lse_gpu.stride(0),
                lse_gpu.stride(1),
                BLOCK_D=128,
                BLOCK_N=block_n,
                num_warps=num_warps,
                num_stages=1,
            )

    if original_device.type == "cuda":
        return output_gpu, lse_gpu
    return output_gpu.to(original_device), lse_gpu.to(original_device)