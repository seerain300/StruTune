import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gqa_paged_prefill_sequence_tile_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    out_ptr,
    lse_ptr,
    sm_scale_log2,
    single_q_len,
    single_kv_len,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kp,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vp,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_qo_indptr,
    stride_kv_indptr,
    stride_kv_indices,
    stride_ot,
    stride_oh,
    stride_od,
    stride_lt,
    stride_lh,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    CONTIGUOUS: tl.constexpr,
    SINGLE_SEQUENCE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    query_tile_id = tl.program_id(1)
    kv_head = tl.program_id(2)

    if SINGLE_SEQUENCE:
        q_start = 0
        q_end = single_q_len
        kv_start = 0
        kv_end = single_kv_len
    elif CONTIGUOUS:
        q_start = tl.load(qo_indptr_ptr + batch_id)
        q_end = tl.load(qo_indptr_ptr + batch_id + 1)
        kv_start = tl.load(kv_indptr_ptr + batch_id)
        kv_end = tl.load(kv_indptr_ptr + batch_id + 1)
    else:
        q_start = tl.load(
            qo_indptr_ptr + batch_id * stride_qo_indptr
        )
        q_end = tl.load(
            qo_indptr_ptr + (batch_id + 1) * stride_qo_indptr
        )
        kv_start = tl.load(
            kv_indptr_ptr + batch_id * stride_kv_indptr
        )
        kv_end = tl.load(
            kv_indptr_ptr + (batch_id + 1) * stride_kv_indptr
        )

    q_len = q_end - q_start
    kv_len = kv_end - kv_start
    tile_q_start = query_tile_id * BLOCK_M

    if tile_q_start >= q_len:
        return

    rows_per_tile: tl.constexpr = BLOCK_M * GQA_RATIO
    offs_r = tl.arange(0, rows_per_tile)
    offs_d = tl.arange(0, HEAD_DIM)

    local_q = tile_q_start + offs_r // GQA_RATIO
    group_head = offs_r % GQA_RATIO
    query_head = kv_head * GQA_RATIO + group_head
    global_q = q_start + local_q
    query_valid = local_q < q_len

    if CONTIGUOUS:
        q_offsets = (
            global_q[:, None] * 4096
            + query_head[:, None] * HEAD_DIM
            + offs_d[None, :]
        )
    else:
        q_offsets = (
            global_q[:, None] * stride_qt
            + query_head[:, None] * stride_qh
            + offs_d[None, :] * stride_qd
        )

    q = tl.load(
        q_ptr + q_offsets,
        mask=query_valid[:, None],
        other=0.0,
    )

    max_visible = kv_len - q_len + local_q + 1
    row_active = query_valid & (max_visible > 0) & (kv_len > 0)

    valid_queries_in_tile = tl.minimum(
        BLOCK_M, q_len - tile_q_start
    )
    scan_len = tl.minimum(
        kv_len,
        tl.maximum(
            kv_len - q_len + tile_q_start + valid_queries_in_tile,
            0,
        ),
    )

    m_i = tl.full(
        (rows_per_tile,), -float("inf"), tl.float32
    )
    l_i = tl.zeros((rows_per_tile,), tl.float32)
    acc = tl.zeros(
        (rows_per_tile, HEAD_DIM), tl.float32
    )

    num_kv_blocks = tl.cdiv(scan_len, BLOCK_N)
    for block_id in range(0, num_kv_blocks):
        offs_n = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_valid = offs_n < scan_len

        if CONTIGUOUS:
            page_ids = tl.load(
                kv_indices_ptr + kv_start + offs_n,
                mask=kv_valid,
                other=0,
            )
            kv_offsets = (
                page_ids[:, None] * 1024
                + kv_head * HEAD_DIM
                + offs_d[None, :]
            )
            k_offsets = kv_offsets
        else:
            page_ids = tl.load(
                kv_indices_ptr
                + (kv_start + offs_n) * stride_kv_indices,
                mask=kv_valid,
                other=0,
            )
            k_offsets = (
                page_ids[:, None] * stride_kp
                + kv_head * stride_kh
                + offs_d[None, :] * stride_kd
            )

        k = tl.load(
            k_ptr + k_offsets,
            mask=kv_valid[:, None],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(k))
        scores *= sm_scale_log2

        causal_mask = (
            row_active[:, None]
            & kv_valid[None, :]
            & (offs_n[None, :] < max_visible[:, None])
        )
        scores = tl.where(
            causal_mask, scores, -float("inf")
        )

        block_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, block_max)
        alpha = tl.where(
            row_active, tl.exp2(m_i - m_new), 0.0
        )
        probabilities = tl.where(
            causal_mask,
            tl.exp2(scores - m_new[:, None]),
            0.0,
        )

        l_i = l_i * alpha + tl.sum(probabilities, axis=1)
        acc *= alpha[:, None]

        if CONTIGUOUS:
            v_offsets = kv_offsets
        else:
            v_offsets = (
                page_ids[:, None] * stride_vp
                + kv_head * stride_vh
                + offs_d[None, :] * stride_vd
            )

        v = tl.load(
            v_ptr + v_offsets,
            mask=kv_valid[:, None],
            other=0.0,
        )

        acc += tl.dot(probabilities.to(tl.bfloat16), v)
        m_i = tl.where(
            row_active, m_new, -float("inf")
        )

    inv_l = tl.where(row_active, 1.0 / l_i, 0.0)
    output = acc * inv_l[:, None]
    lse = tl.where(
        row_active,
        m_i + tl.log2(l_i),
        -float("inf"),
    )

    if CONTIGUOUS:
        out_offsets = (
            global_q[:, None] * 4096
            + query_head[:, None] * HEAD_DIM
            + offs_d[None, :]
        )
        lse_offsets = global_q * 32 + query_head
    else:
        out_offsets = (
            global_q[:, None] * stride_ot
            + query_head[:, None] * stride_oh
            + offs_d[None, :] * stride_od
        )
        lse_offsets = (
            global_q * stride_lt + query_head * stride_lh
        )

    tl.store(
        out_ptr + out_offsets,
        output,
        mask=query_valid[:, None],
    )
    tl.store(
        lse_ptr + lse_offsets,
        lse,
        mask=query_valid,
    )


def _execution_device(tensors):
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
            return tensor.device
    return torch.device("cuda", torch.cuda.current_device())


def _move_tensor_to_device(tensor, device, name):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device == device:
        return tensor
    if tensor.device.type == "cpu":
        return tensor.cuda(device=device)
    return tensor.to(device=device)


@torch.no_grad()
def run(
    q,
    k_cache,
    v_cache,
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale,
):
    tensor_inputs = (
        q,
        k_cache,
        v_cache,
        qo_indptr,
        kv_indptr,
        kv_indices,
    )

    if not torch.cuda.is_available():
        if any(
            isinstance(tensor, torch.Tensor) and tensor.is_cuda
            for tensor in tensor_inputs
        ):
            raise RuntimeError(
                "CUDA is unavailable, but one or more input tensors "
                "are on a CUDA device"
            )
        raise RuntimeError(
            "CUDA is required to execute the Triton attention kernel"
        )

    if not all(
        isinstance(tensor, torch.Tensor)
        for tensor in tensor_inputs
    ):
        raise TypeError(
            "all tensor inputs must be torch.Tensor instances"
        )

    if q.ndim != 3 or q.shape[1] != 32 or q.shape[2] != 128:
        raise ValueError("q must have shape [total_q, 32, 128]")
    if (
        k_cache.ndim != 4
        or tuple(k_cache.shape[1:]) != (1, 8, 128)
    ):
        raise ValueError(
            "k_cache must have shape [num_pages, 1, 8, 128]"
        )
    if (
        v_cache.ndim != 4
        or tuple(v_cache.shape[1:]) != (1, 8, 128)
    ):
        raise ValueError(
            "v_cache must have shape [num_pages, 1, 8, 128]"
        )
    if qo_indptr.ndim != 1 or kv_indptr.ndim != 1:
        raise ValueError(
            "qo_indptr and kv_indptr must be one-dimensional"
        )
    if qo_indptr.shape[0] != kv_indptr.shape[0]:
        raise ValueError(
            "qo_indptr and kv_indptr must have the same length"
        )
    if kv_indices.ndim != 1:
        raise ValueError("kv_indices must be one-dimensional")

    if q.dtype != torch.bfloat16:
        raise TypeError("q must have dtype torch.bfloat16")
    if (
        k_cache.dtype != torch.bfloat16
        or v_cache.dtype != torch.bfloat16
    ):
        raise TypeError(
            "k_cache and v_cache must have dtype torch.bfloat16"
        )
    if (
        qo_indptr.dtype != torch.int32
        or kv_indptr.dtype != torch.int32
    ):
        raise TypeError(
            "qo_indptr and kv_indptr must have dtype torch.int32"
        )
    if kv_indices.dtype != torch.int32:
        raise TypeError("kv_indices must have dtype torch.int32")

    original_q_device = q.device
    execution_device = _execution_device(tensor_inputs)

    with torch.cuda.device(execution_device):
        q_gpu = _move_tensor_to_device(
            q, execution_device, "q"
        )
        k_gpu = _move_tensor_to_device(
            k_cache, execution_device, "k_cache"
        )
        v_gpu = _move_tensor_to_device(
            v_cache, execution_device, "v_cache"
        )
        qo_gpu = _move_tensor_to_device(
            qo_indptr, execution_device, "qo_indptr"
        )
        kv_gpu = _move_tensor_to_device(
            kv_indptr, execution_device, "kv_indptr"
        )
        indices_gpu = _move_tensor_to_device(
            kv_indices, execution_device, "kv_indices"
        )

        if isinstance(sm_scale, torch.Tensor):
            if sm_scale.numel() != 1:
                raise ValueError("sm_scale must be a scalar")
            sm_scale_value = float(sm_scale.detach().item())
        else:
            sm_scale_value = float(sm_scale)

        total_q = q_gpu.shape[0]
        total_kv = indices_gpu.numel()
        batch_size = qo_gpu.shape[0] - 1

        output_gpu = torch.empty_like(q_gpu)
        lse_gpu = torch.empty(
            (total_q, 32),
            dtype=torch.float32,
            device=execution_device,
        )

        if total_q > 0:
            if batch_size <= 0:
                raise ValueError(
                    "nonempty q requires at least one sequence"
                )

            single_sequence = batch_size == 1

            if single_sequence:
                max_q_len = total_q
            elif qo_indptr.device.type == "cpu":
                q_lengths_cpu = qo_indptr[1:] - qo_indptr[:-1]
                max_q_len = int(torch.max(q_lengths_cpu).item())
            else:
                q_lengths = qo_gpu[1:] - qo_gpu[:-1]
                max_q_len = int(torch.max(q_lengths).item())

            if max_q_len <= 0:
                raise ValueError(
                    "nonempty q is inconsistent with qo_indptr"
                )

            block_m = 2 if max_q_len <= 2 else 4
            tiny_average_kv = total_kv <= batch_size * 16
            short_average_kv = total_kv <= batch_size * 128
            block_n = 16 if tiny_average_kv else 32
            num_warps = (
                4
                if block_m == 2 or short_average_kv
                else 8
            )

            contiguous = (
                q_gpu.is_contiguous()
                and k_gpu.is_contiguous()
                and v_gpu.is_contiguous()
                and qo_gpu.is_contiguous()
                and kv_gpu.is_contiguous()
                and indices_gpu.is_contiguous()
                and output_gpu.is_contiguous()
                and lse_gpu.is_contiguous()
            )

            grid = (
                batch_size,
                triton.cdiv(max_q_len, block_m),
                8,
            )

            _gqa_paged_prefill_sequence_tile_kernel[grid](
                q_gpu,
                k_gpu,
                v_gpu,
                qo_gpu,
                kv_gpu,
                indices_gpu,
                output_gpu,
                lse_gpu,
                sm_scale_value * math.log2(math.e),
                total_q,
                total_kv,
                q_gpu.stride(0),
                q_gpu.stride(1),
                q_gpu.stride(2),
                k_gpu.stride(0),
                k_gpu.stride(1),
                k_gpu.stride(2),
                k_gpu.stride(3),
                v_gpu.stride(0),
                v_gpu.stride(1),
                v_gpu.stride(2),
                v_gpu.stride(3),
                qo_gpu.stride(0),
                kv_gpu.stride(0),
                indices_gpu.stride(0),
                output_gpu.stride(0),
                output_gpu.stride(1),
                output_gpu.stride(2),
                lse_gpu.stride(0),
                lse_gpu.stride(1),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                HEAD_DIM=128,
                GQA_RATIO=4,
                CONTIGUOUS=contiguous,
                SINGLE_SEQUENCE=single_sequence,
                num_warps=num_warps,
                num_stages=1,
            )

        if original_q_device.type == "cuda":
            output = output_gpu.to(device=original_q_device)
            lse = lse_gpu.to(device=original_q_device)
        else:
            output = output_gpu.cpu()
            lse = lse_gpu.cpu()

    return output, lse