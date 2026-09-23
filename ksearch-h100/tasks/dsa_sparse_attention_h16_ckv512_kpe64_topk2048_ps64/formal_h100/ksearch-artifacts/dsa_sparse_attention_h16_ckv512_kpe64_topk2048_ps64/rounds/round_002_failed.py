# solution=GPT-5.6-Sol_dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64_triton_optimized_r2 score=-1.0 passed=False
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_sparse_attention_kernel(
    q_nope_ptr,
    q_pe_ptr,
    ckv_cache_ptr,
    kpe_cache_ptr,
    sparse_indices_ptr,
    output_ptr,
    lse_ptr,
    sm_scale,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    HEAD_DIM_KPE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)

    head_offsets = tl.arange(0, NUM_HEADS)
    dim_offsets = tl.arange(0, BLOCK_D)

    q_nope_base = token_idx * NUM_HEADS * HEAD_DIM_CKV
    q_pe_base = token_idx * NUM_HEADS * HEAD_DIM_KPE

    running_max = tl.full([NUM_HEADS], -float("inf"), tl.float32)
    running_sum = tl.zeros([NUM_HEADS], tl.float32)

    acc0 = tl.zeros([NUM_HEADS, BLOCK_D], tl.float32)
    acc1 = tl.zeros([NUM_HEADS, BLOCK_D], tl.float32)
    acc2 = tl.zeros([NUM_HEADS, BLOCK_D], tl.float32)
    acc3 = tl.zeros([NUM_HEADS, BLOCK_D], tl.float32)
    acc4 = tl.zeros([NUM_HEADS, BLOCK_D], tl.float32)
    acc5 = tl.zeros([NUM_HEADS, BLOCK_D], tl.float32)
    acc6 = tl.zeros([NUM_HEADS, BLOCK_D], tl.float32)
    acc7 = tl.zeros([NUM_HEADS, BLOCK_D], tl.float32)

    scaled_log2 = sm_scale * 1.4426950408889634

    for sparse_start in tl.range(0, TOPK, BLOCK_N):
        sparse_offsets = sparse_start + tl.arange(0, BLOCK_N)
        cache_indices = tl.load(
            sparse_indices_ptr + token_idx * TOPK + sparse_offsets
        )
        valid = cache_indices != -1
        safe_indices = tl.where(valid, cache_indices, 0)

        scores = tl.zeros([NUM_HEADS, BLOCK_N], tl.float32)

        for dim_start in tl.static_range(0, HEAD_DIM_CKV, BLOCK_D):
            current_dims = dim_start + dim_offsets

            query = tl.load(
                q_nope_ptr
                + q_nope_base
                + head_offsets[:, None] * HEAD_DIM_CKV
                + current_dims[None, :]
            )
            keys = tl.load(
                ckv_cache_ptr
                + safe_indices[:, None] * HEAD_DIM_CKV
                + current_dims[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            scores += tl.dot(query, tl.trans(keys))

        positional_query = tl.load(
            q_pe_ptr
            + q_pe_base
            + head_offsets[:, None] * HEAD_DIM_KPE
            + dim_offsets[None, :]
        )
        positional_keys = tl.load(
            kpe_cache_ptr
            + safe_indices[:, None] * HEAD_DIM_KPE
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        scores += tl.dot(positional_query, tl.trans(positional_keys))

        scores *= scaled_log2
        scores = tl.where(valid[None, :], scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        old_scale = tl.where(
            running_max == -float("inf"),
            0.0,
            tl.exp2(running_max - new_max),
        )

        probabilities = tl.where(
            valid[None, :],
            tl.exp2(scores - new_max[:, None]),
            0.0,
        )
        probability_sum = tl.sum(probabilities, axis=1)
        probabilities_bf16 = probabilities.to(tl.bfloat16)

        acc0 *= old_scale[:, None]
        acc1 *= old_scale[:, None]
        acc2 *= old_scale[:, None]
        acc3 *= old_scale[:, None]
        acc4 *= old_scale[:, None]
        acc5 *= old_scale[:, None]
        acc6 *= old_scale[:, None]
        acc7 *= old_scale[:, None]

        values0 = tl.load(
            ckv_cache_ptr
            + safe_indices[:, None] * HEAD_DIM_CKV
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        acc0 += tl.dot(probabilities_bf16, values0)

        values1 = tl.load(
            ckv_cache_ptr
            + safe_indices[:, None] * HEAD_DIM_CKV
            + BLOCK_D
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        acc1 += tl.dot(probabilities_bf16, values1)

        values2 = tl.load(
            ckv_cache_ptr
            + safe_indices[:, None] * HEAD_DIM_CKV
            + 2 * BLOCK_D
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        acc2 += tl.dot(probabilities_bf16, values2)

        values3 = tl.load(
            ckv_cache_ptr
            + safe_indices[:, None] * HEAD_DIM_CKV
            + 3 * BLOCK_D
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        acc3 += tl.dot(probabilities_bf16, values3)

        values4 = tl.load(
            ckv_cache_ptr
            + safe_indices[:, None] * HEAD_DIM_CKV
            + 4 * BLOCK_D
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        acc4 += tl.dot(probabilities_bf16, values4)

        values5 = tl.load(
            ckv_cache_ptr
            + safe_indices[:, None] * HEAD_DIM_CKV
            + 5 * BLOCK_D
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        acc5 += tl.dot(probabilities_bf16, values5)

        values6 = tl.load(
            ckv_cache_ptr
            + safe_indices[:, None] * HEAD_DIM_CKV
            + 6 * BLOCK_D
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        acc6 += tl.dot(probabilities_bf16, values6)

        values7 = tl.load(
            ckv_cache_ptr
            + safe_indices[:, None] * HEAD_DIM_CKV
            + 7 * BLOCK_D
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        acc7 += tl.dot(probabilities_bf16, values7)

        running_sum = running_sum * old_scale + probability_sum
        running_max = new_max

    nonempty = running_sum > 0.0
    denominator = tl.where(nonempty, running_sum, 1.0)
    inverse_denominator = 1.0 / denominator

    output_base = token_idx * NUM_HEADS * HEAD_DIM_CKV
    output_offsets = (
        output_base
        + head_offsets[:, None] * HEAD_DIM_CKV
        + dim_offsets[None, :]
    )

    tl.store(
        output_ptr + output_offsets,
        tl.where(
            nonempty[:, None],
            acc0 * inverse_denominator[:, None],
            0.0,
        ),
    )
    tl.store(
        output_ptr + output_offsets + BLOCK_D,
        tl.where(
            nonempty[:, None],
            acc1 * inverse_denominator[:, None],
            0.0,
        ),
    )
    tl.store(
        output_ptr + output_offsets + 2 * BLOCK_D,
        tl.where(
            nonempty[:, None],
            acc2 * inverse_denominator[:, None],
            0.0,
        ),
    )
    tl.store(
        output_ptr + output_offsets + 3 * BLOCK_D,
        tl.where(
            nonempty[:, None],
            acc3 * inverse_denominator[:, None],
            0.0,
        ),
    )
    tl.store(
        output_ptr + output_offsets + 4 * BLOCK_D,
        tl.where(
            nonempty[:, None],
            acc4 * inverse_denominator[:, None],
            0.0,
        ),
    )
    tl.store(
        output_ptr + output_offsets + 5 * BLOCK_D,
        tl.where(
            nonempty[:, None],
            acc5 * inverse_denominator[:, None],
            0.0,
        ),
    )
    tl.store(
        output_ptr + output_offsets + 6 * BLOCK_D,
        tl.where(
            nonempty[:, None],
            acc6 * inverse_denominator[:, None],
            0.0,
        ),
    )
    tl.store(
        output_ptr + output_offsets + 7 * BLOCK_D,
        tl.where(
            nonempty[:, None],
            acc7 * inverse_denominator[:, None],
            0.0,
        ),
    )

    lse = tl.where(
        nonempty,
        running_max + tl.log2(denominator),
        -float("inf"),
    )
    tl.store(
        lse_ptr + token_idx * NUM_HEADS + head_offsets,
        lse,
    )


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    tensors = {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
        "sparse_indices": sparse_indices,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the Triton attention kernel")

    if q_nope.ndim != 3 or q_nope.shape[1:] != (16, 512):
        raise ValueError("q_nope must have shape [num_tokens, 16, 512]")
    if q_pe.shape != (q_nope.shape[0], 16, 64):
        raise ValueError("q_pe must have shape [num_tokens, 16, 64]")
    if ckv_cache.ndim != 3 or ckv_cache.shape[1:] != (64, 512):
        raise ValueError("ckv_cache must have shape [num_pages, 64, 512]")
    if kpe_cache.shape != (ckv_cache.shape[0], 64, 64):
        raise ValueError("kpe_cache must have shape [num_pages, 64, 64]")
    if sparse_indices.shape != (q_nope.shape[0], 2048):
        raise ValueError("sparse_indices must have shape [num_tokens, 2048]")

    if q_nope.dtype != torch.bfloat16:
        raise TypeError("q_nope must have dtype torch.bfloat16")
    if q_pe.dtype != torch.bfloat16:
        raise TypeError("q_pe must have dtype torch.bfloat16")
    if ckv_cache.dtype != torch.bfloat16:
        raise TypeError("ckv_cache must have dtype torch.bfloat16")
    if kpe_cache.dtype != torch.bfloat16:
        raise TypeError("kpe_cache must have dtype torch.bfloat16")
    if sparse_indices.dtype != torch.int32:
        raise TypeError("sparse_indices must have dtype torch.int32")

    if isinstance(sm_scale, torch.Tensor):
        if sm_scale.numel() != 1:
            raise ValueError("sm_scale must be a scalar")
        sm_scale_value = float(sm_scale.detach().cpu().item())
    else:
        sm_scale_value = float(sm_scale)

    original_device = q_nope.device
    if q_nope.is_cuda:
        target_device = q_nope.device
    else:
        target_device = torch.device("cuda", torch.cuda.current_device())

    with torch.cuda.device(target_device):
        q_nope_gpu = q_nope.to(
            device=target_device, dtype=torch.bfloat16
        ).contiguous()
        q_pe_gpu = q_pe.to(
            device=target_device, dtype=torch.bfloat16
        ).contiguous()
        ckv_cache_gpu = ckv_cache.to(
            device=target_device, dtype=torch.bfloat16
        ).contiguous()
        kpe_cache_gpu = kpe_cache.to(
            device=target_device, dtype=torch.bfloat16
        ).contiguous()
        sparse_indices_gpu = sparse_indices.to(
            device=target_device, dtype=torch.int32
        ).contiguous()

        num_tokens = q_nope_gpu.shape[0]
        output_gpu = torch.empty_like(q_nope_gpu)
        lse_gpu = torch.empty(
            (num_tokens, 16),
            dtype=torch.float32,
            device=target_device,
        )

        _fused_sparse_attention_kernel[(num_tokens,)](
            q_nope_gpu,
            q_pe_gpu,
            ckv_cache_gpu,
            kpe_cache_gpu,
            sparse_indices_gpu,
            output_gpu,
            lse_gpu,
            sm_scale_value,
            NUM_HEADS=16,
            HEAD_DIM_CKV=512,
            HEAD_DIM_KPE=64,
            TOPK=2048,
            BLOCK_N=32,
            BLOCK_D=64,
            num_warps=8,
            num_stages=2,
        )

    return (
        output_gpu.to(device=original_device),
        lse_gpu.to(device=original_device),
    )