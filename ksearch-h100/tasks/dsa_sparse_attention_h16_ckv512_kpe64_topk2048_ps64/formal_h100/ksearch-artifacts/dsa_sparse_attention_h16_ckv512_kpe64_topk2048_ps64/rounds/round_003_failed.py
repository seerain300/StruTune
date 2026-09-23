# solution=GPT-5.6-Sol_dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64_triton_optimized_r3 score=-1.0 passed=False
I’m keeping the two-kernel structure aligned with the supplied base and targeting the reduction math first. The main change is to use base-2 logits and `exp2`, which matches the required LSE representation and removes repeated natural-log/exponential conversion work while preserving the online-softmax handling of invalid slots.import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_attention_logits_kernel(
    q_nope,
    q_pe,
    ckv_cache,
    kpe_cache,
    sparse_indices,
    logits_buffer,
    sm_scale,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    HEAD_DIM_KPE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    slot_block = tl.program_id(0)
    token_idx = tl.program_id(1)

    slot_offsets = slot_block * BLOCK_N + tl.arange(0, BLOCK_N)
    head_offsets = tl.arange(0, NUM_HEADS)

    index_base = sparse_indices + token_idx * TOPK
    selected = tl.load(index_base + slot_offsets)
    valid = selected >= 0
    safe_selected = tl.where(valid, selected, 0)

    qn_base = q_nope + token_idx * NUM_HEADS * HEAD_DIM_CKV
    logits = tl.zeros((BLOCK_N, NUM_HEADS), dtype=tl.float32)

    for dim_start in range(0, HEAD_DIM_CKV, 64):
        dim_offsets = dim_start + tl.arange(0, 64)

        keys = tl.load(
            ckv_cache
            + safe_selected[:, None] * HEAD_DIM_CKV
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        queries = tl.load(
            qn_base
            + dim_offsets[:, None]
            + head_offsets[None, :] * HEAD_DIM_CKV
        )
        logits += tl.dot(keys, queries)

    pe_offsets = tl.arange(0, HEAD_DIM_KPE)
    pe_keys = tl.load(
        kpe_cache
        + safe_selected[:, None] * HEAD_DIM_KPE
        + pe_offsets[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    pe_queries = tl.load(
        q_pe
        + token_idx * NUM_HEADS * HEAD_DIM_KPE
        + pe_offsets[:, None]
        + head_offsets[None, :] * HEAD_DIM_KPE
    )
    logits += tl.dot(pe_keys, pe_queries)

    logits *= sm_scale * 1.4426950408889634
    logits = tl.where(valid[:, None], logits, -float("inf"))

    logits_offsets = (
        token_idx * NUM_HEADS * TOPK
        + head_offsets[:, None] * TOPK
        + slot_offsets[None, :]
    )
    tl.store(logits_buffer + logits_offsets, tl.trans(logits))


@triton.jit
def _sparse_attention_output_kernel(
    logits_buffer,
    ckv_cache,
    sparse_indices,
    output,
    lse,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    d_tile = tl.program_id(0)
    token_idx = tl.program_id(1)

    head_offsets = tl.arange(0, NUM_HEADS)
    out_offsets = d_tile * BLOCK_D + tl.arange(0, BLOCK_D)
    out_mask = out_offsets < HEAD_DIM_CKV

    logits_base = logits_buffer + token_idx * NUM_HEADS * TOPK
    index_base = sparse_indices + token_idx * TOPK

    running_max = tl.full((NUM_HEADS,), -float("inf"), tl.float32)
    running_sum = tl.zeros((NUM_HEADS,), dtype=tl.float32)
    accumulator = tl.zeros((NUM_HEADS, BLOCK_D), dtype=tl.float32)

    for slot_start in range(0, TOPK, BLOCK_N):
        slot_offsets = slot_start + tl.arange(0, BLOCK_N)

        selected = tl.load(index_base + slot_offsets)
        valid = selected >= 0
        safe_selected = tl.where(valid, selected, 0)

        logits = tl.load(
            logits_base
            + head_offsets[:, None] * TOPK
            + slot_offsets[None, :]
        )
        logits = tl.where(valid[None, :], logits, -float("inf"))

        block_max = tl.max(logits, axis=1)
        block_has_values = tl.sum(valid.to(tl.int32), axis=0) > 0
        new_max = tl.where(
            block_has_values,
            tl.maximum(running_max, block_max),
            running_max,
        )

        old_scale = tl.where(
            running_sum > 0.0,
            tl.exp2(running_max - new_max),
            0.0,
        )
        weights = tl.where(
            valid[None, :],
            tl.exp2(logits - new_max[:, None]),
            0.0,
        )
        block_sum = tl.sum(weights, axis=1)

        values = tl.load(
            ckv_cache
            + safe_selected[:, None] * HEAD_DIM_CKV
            + out_offsets[None, :],
            mask=valid[:, None] & out_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        accumulator = accumulator * old_scale[:, None] + tl.dot(
            weights,
            values,
        )
        running_sum = running_sum * old_scale + block_sum
        running_max = new_max

    normalized = tl.where(
        running_sum[:, None] > 0.0,
        accumulator / running_sum[:, None],
        0.0,
    )

    output_offsets = (
        token_idx * NUM_HEADS * HEAD_DIM_CKV
        + head_offsets[:, None] * HEAD_DIM_CKV
        + out_offsets[None, :]
    )
    tl.store(
        output + output_offsets,
        normalized,
        mask=out_mask[None, :],
    )

    if d_tile == 0:
        lse_values = tl.where(
            running_sum > 0.0,
            running_max + tl.log(running_sum) * 1.4426950408889634,
            -float("inf"),
        )
        tl.store(
            lse + token_idx * NUM_HEADS + head_offsets,
            lse_values,
        )


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "dsa sparse attention requires a CUDA-capable GPU, but CUDA is unavailable"
        )

    if q_nope.ndim != 3 or q_nope.shape[1:] != (16, 512):
        raise ValueError("q_nope must have shape [num_tokens, 16, 512]")
    if q_pe.ndim != 3 or q_pe.shape != (q_nope.shape[0], 16, 64):
        raise ValueError("q_pe must have shape [num_tokens, 16, 64]")
    if ckv_cache.ndim != 3 or ckv_cache.shape[1:] != (64, 512):
        raise ValueError("ckv_cache must have shape [num_pages, 64, 512]")
    if kpe_cache.ndim != 3 or kpe_cache.shape != (
        ckv_cache.shape[0],
        64,
        64,
    ):
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

    original_device = q_nope.device
    cuda_device = (
        q_nope.device
        if q_nope.is_cuda
        else torch.device("cuda", torch.cuda.current_device())
    )

    def to_cuda_contiguous(tensor):
        if tensor.is_cuda:
            return tensor.to(device=cuda_device).contiguous()
        return tensor.cuda(device=cuda_device).contiguous()

    q_nope_cuda = to_cuda_contiguous(q_nope)
    q_pe_cuda = to_cuda_contiguous(q_pe)
    ckv_cache_cuda = to_cuda_contiguous(ckv_cache)
    kpe_cache_cuda = to_cuda_contiguous(kpe_cache)
    sparse_indices_cuda = to_cuda_contiguous(sparse_indices)

    if isinstance(sm_scale, torch.Tensor):
        if sm_scale.numel() != 1:
            raise ValueError("sm_scale must be a scalar")
        sm_scale_value = float(sm_scale.detach().item())
    else:
        sm_scale_value = float(sm_scale)

    num_tokens = q_nope.shape[0]

    with torch.cuda.device(cuda_device):
        logits_buffer = torch.empty(
            (num_tokens, 16, 2048),
            dtype=torch.float32,
            device=cuda_device,
        )
        output_cuda = torch.empty(
            (num_tokens, 16, 512),
            dtype=torch.bfloat16,
            device=cuda_device,
        )
        lse_cuda = torch.empty(
            (num_tokens, 16),
            dtype=torch.float32,
            device=cuda_device,
        )

        logits_block_n = 32
        _sparse_attention_logits_kernel[
            (triton.cdiv(2048, logits_block_n), num_tokens)
        ](
            q_nope_cuda,
            q_pe_cuda,
            ckv_cache_cuda,
            kpe_cache_cuda,
            sparse_indices_cuda,
            logits_buffer,
            sm_scale_value,
            NUM_HEADS=16,
            HEAD_DIM_CKV=512,
            HEAD_DIM_KPE=64,
            TOPK=2048,
            BLOCK_N=logits_block_n,
            num_warps=4,
            num_stages=2,
        )

        output_block_n = 32
        output_block_d = 32
        _sparse_attention_output_kernel[
            (triton.cdiv(512, output_block_d), num_tokens)
        ](
            logits_buffer,
            ckv_cache_cuda,
            sparse_indices_cuda,
            output_cuda,
            lse_cuda,
            NUM_HEADS=16,
            HEAD_DIM_CKV=512,
            TOPK=2048,
            BLOCK_N=output_block_n,
            BLOCK_D=output_block_d,
            num_warps=8,
            num_stages=2,
        )

    if original_device.type == "cuda" and original_device == cuda_device:
        return output_cuda, lse_cuda

    return (
        output_cuda.to(device=original_device),
        lse_cuda.to(device=original_device),
    )