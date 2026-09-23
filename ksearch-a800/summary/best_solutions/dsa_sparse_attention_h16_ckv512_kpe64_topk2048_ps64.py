# task: dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64
# bench: FlashInfer | batch: formal_20260914
# final eval (official evaluator, full workloads): valid=True pass=23/23 geomean=20.720x
# feedback best (5-workload sample during search): 20.999x
# torch fallback audit: 干净 (-)
# tokens: 1,624,052

import torch
import triton
import triton.language as tl


@triton.jit
def _token_complete_attention_kernel(
    q_nope_ptr,
    q_pe_ptr,
    ckv_ptr,
    kpe_ptr,
    indices_ptr,
    output_ptr,
    lse_ptr,
    sm_scale,
    stride_qn_t,
    stride_qn_h,
    stride_qn_d,
    stride_qp_t,
    stride_qp_h,
    stride_qp_d,
    stride_ckv_p,
    stride_ckv_s,
    stride_ckv_d,
    stride_kpe_p,
    stride_kpe_s,
    stride_kpe_d,
    stride_idx_t,
    stride_idx_k,
    stride_out_t,
    stride_out_h,
    stride_out_d,
    stride_lse_t,
    stride_lse_h,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    HEAD_DIM_KPE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FLAT_CACHE: tl.constexpr,
):
    token_idx = tl.program_id(0)

    head_offsets = tl.arange(0, NUM_HEADS)
    ckv_offsets = tl.arange(0, HEAD_DIM_CKV)
    kpe_offsets = tl.arange(0, HEAD_DIM_KPE)

    q_nope = tl.load(
        q_nope_ptr
        + token_idx * stride_qn_t
        + head_offsets[:, None] * stride_qn_h
        + ckv_offsets[None, :] * stride_qn_d
    )
    q_pe = tl.load(
        q_pe_ptr
        + token_idx * stride_qp_t
        + head_offsets[:, None] * stride_qp_h
        + kpe_offsets[None, :] * stride_qp_d
    )

    running_max = tl.full((NUM_HEADS,), -float("inf"), tl.float32)
    running_sum = tl.zeros((NUM_HEADS,), tl.float32)
    accumulator = tl.zeros(
        (NUM_HEADS, HEAD_DIM_CKV),
        dtype=tl.float32,
    )

    scaled_log2 = sm_scale * 1.4426950408889634

    for start_n in tl.range(0, TOPK, BLOCK_N, num_stages=2):
        selected_offsets = start_n + tl.arange(0, BLOCK_N)
        selected_indices = tl.load(
            indices_ptr
            + token_idx * stride_idx_t
            + selected_offsets * stride_idx_k
        )
        valid = selected_indices >= 0

        if FLAT_CACHE:
            ckv_row_offsets = selected_indices * stride_ckv_s
            kpe_row_offsets = selected_indices * stride_kpe_s
        else:
            page_indices = selected_indices // PAGE_SIZE
            page_offsets = selected_indices - page_indices * PAGE_SIZE
            ckv_row_offsets = (
                page_indices * stride_ckv_p
                + page_offsets * stride_ckv_s
            )
            kpe_row_offsets = (
                page_indices * stride_kpe_p
                + page_offsets * stride_kpe_s
            )

        ckv = tl.load(
            ckv_ptr
            + ckv_row_offsets[:, None]
            + ckv_offsets[None, :] * stride_ckv_d,
            mask=valid[:, None],
            other=0.0,
        )
        kpe = tl.load(
            kpe_ptr
            + kpe_row_offsets[:, None]
            + kpe_offsets[None, :] * stride_kpe_d,
            mask=valid[:, None],
            other=0.0,
        )

        scores = tl.dot(
            q_nope,
            tl.trans(ckv),
            out_dtype=tl.float32,
        )
        scores += tl.dot(
            q_pe,
            tl.trans(kpe),
            out_dtype=tl.float32,
        )
        scores *= scaled_log2
        scores = tl.where(
            valid[None, :],
            scores,
            -float("inf"),
        )

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        has_values = new_max != -float("inf")

        old_scale = tl.where(
            has_values,
            tl.exp2(running_max - new_max),
            0.0,
        )
        probabilities = tl.where(
            valid[None, :] & has_values[:, None],
            tl.exp2(scores - new_max[:, None]),
            0.0,
        )

        accumulator *= old_scale[:, None]
        accumulator += tl.dot(
            probabilities.to(tl.bfloat16),
            ckv,
            out_dtype=tl.float32,
        )
        running_sum = (
            running_sum * old_scale
            + tl.sum(probabilities, axis=1)
        )
        running_max = new_max

    nonempty = running_sum > 0.0
    inv_sum = tl.where(nonempty, 1.0 / running_sum, 0.0)
    normalized = accumulator * inv_sum[:, None]
    lse = tl.where(
        nonempty,
        running_max + tl.log2(running_sum),
        -float("inf"),
    )

    tl.store(
        output_ptr
        + token_idx * stride_out_t
        + head_offsets[:, None] * stride_out_h
        + ckv_offsets[None, :] * stride_out_d,
        normalized,
    )
    tl.store(
        lse_ptr
        + token_idx * stride_lse_t
        + head_offsets * stride_lse_h,
        lse,
    )


def run(
    q_nope,
    q_pe,
    ckv_cache,
    kpe_cache,
    sparse_indices,
    sm_scale,
):
    tensor_arguments = {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
        "sparse_indices": sparse_indices,
    }

    for name, value in tensor_arguments.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    if q_nope.ndim != 3 or q_nope.shape[1:] != (16, 512):
        raise ValueError("q_nope must have shape [num_tokens, 16, 512]")

    num_tokens = q_nope.shape[0]

    if q_pe.ndim != 3 or q_pe.shape != (num_tokens, 16, 64):
        raise ValueError("q_pe must have shape [num_tokens, 16, 64]")
    if ckv_cache.ndim != 3 or ckv_cache.shape[1:] != (64, 512):
        raise ValueError("ckv_cache must have shape [num_pages, 64, 512]")
    if kpe_cache.ndim != 3 or kpe_cache.shape != (
        ckv_cache.shape[0],
        64,
        64,
    ):
        raise ValueError("kpe_cache must have shape [num_pages, 64, 64]")
    if sparse_indices.ndim != 2 or sparse_indices.shape != (
        num_tokens,
        2048,
    ):
        raise ValueError(
            "sparse_indices must have shape [num_tokens, 2048]"
        )

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

    if not torch.cuda.is_available():
        if any(value.is_cuda for value in tensor_arguments.values()):
            raise RuntimeError(
                "CUDA tensors were provided, but CUDA is not available"
            )
        raise RuntimeError("CUDA is required to execute this Triton kernel")

    original_output_device = q_nope.device

    if q_nope.is_cuda:
        execution_device = q_nope.device
    else:
        cuda_devices = [
            value.device
            for value in tensor_arguments.values()
            if value.is_cuda
        ]
        execution_device = (
            cuda_devices[0]
            if cuda_devices
            else torch.device("cuda", torch.cuda.current_device())
        )

    def move_to_execution_device(tensor):
        if tensor.device == execution_device:
            return tensor
        if tensor.is_cuda:
            return tensor.to(execution_device)
        return tensor.cuda(device=execution_device)

    q_nope_gpu = move_to_execution_device(q_nope)
    q_pe_gpu = move_to_execution_device(q_pe)
    ckv_cache_gpu = move_to_execution_device(ckv_cache)
    kpe_cache_gpu = move_to_execution_device(kpe_cache)
    sparse_indices_gpu = move_to_execution_device(sparse_indices)

    if isinstance(sm_scale, torch.Tensor):
        if sm_scale.numel() != 1:
            raise ValueError("sm_scale must be a scalar")
        sm_scale_value = float(sm_scale.detach().cpu().item())
    else:
        sm_scale_value = float(sm_scale)

    flat_cache = (
        ckv_cache_gpu.stride(0) == 64 * ckv_cache_gpu.stride(1)
        and kpe_cache_gpu.stride(0) == 64 * kpe_cache_gpu.stride(1)
    )

    with torch.cuda.device(execution_device):
        output_gpu = torch.empty(
            (num_tokens, 16, 512),
            dtype=torch.bfloat16,
            device=execution_device,
        )
        lse_gpu = torch.empty(
            (num_tokens, 16),
            dtype=torch.float32,
            device=execution_device,
        )

        if num_tokens > 0:
            _token_complete_attention_kernel[(num_tokens,)](
                q_nope_gpu,
                q_pe_gpu,
                ckv_cache_gpu,
                kpe_cache_gpu,
                sparse_indices_gpu,
                output_gpu,
                lse_gpu,
                sm_scale_value,
                q_nope_gpu.stride(0),
                q_nope_gpu.stride(1),
                q_nope_gpu.stride(2),
                q_pe_gpu.stride(0),
                q_pe_gpu.stride(1),
                q_pe_gpu.stride(2),
                ckv_cache_gpu.stride(0),
                ckv_cache_gpu.stride(1),
                ckv_cache_gpu.stride(2),
                kpe_cache_gpu.stride(0),
                kpe_cache_gpu.stride(1),
                kpe_cache_gpu.stride(2),
                sparse_indices_gpu.stride(0),
                sparse_indices_gpu.stride(1),
                output_gpu.stride(0),
                output_gpu.stride(1),
                output_gpu.stride(2),
                lse_gpu.stride(0),
                lse_gpu.stride(1),
                NUM_HEADS=16,
                HEAD_DIM_CKV=512,
                HEAD_DIM_KPE=64,
                PAGE_SIZE=64,
                TOPK=2048,
                BLOCK_N=64,
                FLAT_CACHE=flat_cache,
                num_warps=4,
            )

    if output_gpu.device != original_output_device:
        output = output_gpu.to(original_output_device)
        lse = lse_gpu.to(original_output_device)
    else:
        output = output_gpu
        lse = lse_gpu

    return output, lse