# task: gqa_paged_decode_h32_kv8_d128_ps1
# bench: FlashInfer | batch: formal_20260914
# final eval (official evaluator, full workloads): valid=True pass=48/48 geomean=124.867x
# feedback best (5-workload sample during search): 178.854x
# torch fallback audit: 干净 (-)
# tokens: 3,182,294

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _single_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    indptr_ptr,
    indices_ptr,
    out_ptr,
    lse_ptr,
    sm_scale_log2: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kp: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vp: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_indptr: tl.constexpr,
    stride_indices: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_index = pid // 8
    kv_head = pid % 8
    first_q_head = kv_head * 4
    offsets_d = tl.arange(0, HEAD_DIM)

    page_start = tl.load(
        indptr_ptr + batch_index * stride_indptr
    ).to(tl.int32)
    page_end = tl.load(
        indptr_ptr + (batch_index + 1) * stride_indptr
    ).to(tl.int32)
    is_nonempty = page_start < page_end

    page_id = tl.load(
        indices_ptr + page_start * stride_indices,
        mask=is_nonempty,
        other=0,
    ).to(tl.int32)

    q_base = (
        q_ptr
        + batch_index * stride_qb
        + first_q_head * stride_qh
        + offsets_d * stride_qd
    )
    q0 = tl.load(q_base).to(tl.float32) * sm_scale_log2
    q1 = tl.load(q_base + stride_qh).to(tl.float32) * sm_scale_log2
    q2 = tl.load(q_base + 2 * stride_qh).to(tl.float32) * sm_scale_log2
    q3 = tl.load(q_base + 3 * stride_qh).to(tl.float32) * sm_scale_log2

    k_offsets = (
        page_id * stride_kp
        + kv_head * stride_kh
        + offsets_d * stride_kd
    )
    k = tl.load(
        k_ptr + k_offsets,
        mask=is_nonempty,
        other=0.0,
    ).to(tl.float32)

    score0 = tl.sum(k * q0, axis=0)
    score1 = tl.sum(k * q1, axis=0)
    score2 = tl.sum(k * q2, axis=0)
    score3 = tl.sum(k * q3, axis=0)

    v_offsets = (
        page_id * stride_vp
        + kv_head * stride_vh
        + offsets_d * stride_vd
    )
    value = tl.load(
        v_ptr + v_offsets,
        mask=is_nonempty,
        other=0.0,
    )

    out_base = (
        out_ptr
        + batch_index * 4096
        + first_q_head * HEAD_DIM
        + offsets_d
    )
    tl.store(out_base, value)
    tl.store(out_base + HEAD_DIM, value)
    tl.store(out_base + 2 * HEAD_DIM, value)
    tl.store(out_base + 3 * HEAD_DIM, value)

    lse_base = lse_ptr + batch_index * 32 + first_q_head
    empty_lse = -float("inf")
    tl.store(lse_base, tl.where(is_nonempty, score0, empty_lse))
    tl.store(lse_base + 1, tl.where(is_nonempty, score1, empty_lse))
    tl.store(lse_base + 2, tl.where(is_nonempty, score2, empty_lse))
    tl.store(lse_base + 3, tl.where(is_nonempty, score3, empty_lse))


@triton.jit
def _short_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    indptr_ptr,
    indices_ptr,
    out_ptr,
    lse_ptr,
    sm_scale_log2: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kp: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vp: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_indptr: tl.constexpr,
    stride_indices: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FIXED_LENGTH: tl.constexpr,
    SERIAL_ACCUM: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_index = pid // 8
    kv_head = pid % 8
    first_q_head = kv_head * 4

    offsets_d = tl.arange(0, HEAD_DIM)
    offsets_n = tl.arange(0, BLOCK_N)

    q_base = (
        q_ptr
        + batch_index * stride_qb
        + first_q_head * stride_qh
        + offsets_d * stride_qd
    )
    q0 = tl.load(q_base).to(tl.float32) * sm_scale_log2
    q1 = tl.load(q_base + stride_qh).to(tl.float32) * sm_scale_log2
    q2 = tl.load(q_base + 2 * stride_qh).to(tl.float32) * sm_scale_log2
    q3 = tl.load(q_base + 3 * stride_qh).to(tl.float32) * sm_scale_log2

    page_start = tl.load(
        indptr_ptr + batch_index * stride_indptr
    ).to(tl.int32)

    if FIXED_LENGTH > 0:
        valid = offsets_n < FIXED_LENGTH
        is_nonempty = True
    else:
        page_end = tl.load(
            indptr_ptr + (batch_index + 1) * stride_indptr
        ).to(tl.int32)
        valid = page_start + offsets_n < page_end
        is_nonempty = page_start < page_end

    positions = page_start + offsets_n
    page_ids = tl.load(
        indices_ptr + positions * stride_indices,
        mask=valid,
        other=0,
    ).to(tl.int32)

    k_offsets = (
        page_ids[:, None] * stride_kp
        + kv_head * stride_kh
        + offsets_d[None, :] * stride_kd
    )
    k = tl.load(
        k_ptr + k_offsets,
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)

    score0 = tl.sum(k * q0[None, :], axis=1)
    score1 = tl.sum(k * q1[None, :], axis=1)
    score2 = tl.sum(k * q2[None, :], axis=1)
    score3 = tl.sum(k * q3[None, :], axis=1)

    score0 = tl.where(valid, score0, -float("inf"))
    score1 = tl.where(valid, score1, -float("inf"))
    score2 = tl.where(valid, score2, -float("inf"))
    score3 = tl.where(valid, score3, -float("inf"))

    max0 = tl.max(score0, axis=0)
    max1 = tl.max(score1, axis=0)
    max2 = tl.max(score2, axis=0)
    max3 = tl.max(score3, axis=0)

    max0 = tl.where(is_nonempty, max0, 0.0)
    max1 = tl.where(is_nonempty, max1, 0.0)
    max2 = tl.where(is_nonempty, max2, 0.0)
    max3 = tl.where(is_nonempty, max3, 0.0)

    weight0 = tl.where(valid, tl.exp2(score0 - max0), 0.0)
    weight1 = tl.where(valid, tl.exp2(score1 - max1), 0.0)
    weight2 = tl.where(valid, tl.exp2(score2 - max2), 0.0)
    weight3 = tl.where(valid, tl.exp2(score3 - max3), 0.0)

    sum0 = tl.sum(weight0, axis=0)
    sum1 = tl.sum(weight1, axis=0)
    sum2 = tl.sum(weight2, axis=0)
    sum3 = tl.sum(weight3, axis=0)

    safe_sum0 = tl.where(is_nonempty, sum0, 1.0)
    safe_sum1 = tl.where(is_nonempty, sum1, 1.0)
    safe_sum2 = tl.where(is_nonempty, sum2, 1.0)
    safe_sum3 = tl.where(is_nonempty, sum3, 1.0)

    v_offsets = (
        page_ids[:, None] * stride_vp
        + kv_head * stride_vh
        + offsets_d[None, :] * stride_vd
    )
    v = tl.load(
        v_ptr + v_offsets,
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)

    out_base = (
        out_ptr
        + batch_index * 4096
        + first_q_head * HEAD_DIM
        + offsets_d
    )
    lse_base = lse_ptr + batch_index * 32 + first_q_head

    if SERIAL_ACCUM:
        acc0 = tl.sum(v * weight0[:, None], axis=0)
        out0 = tl.where(is_nonempty, acc0 / safe_sum0, 0.0)
        lse0 = tl.where(
            is_nonempty,
            max0 + tl.log2(safe_sum0),
            -float("inf"),
        )
        tl.store(out_base, out0)
        tl.store(lse_base, lse0)

        acc1 = tl.sum(v * weight1[:, None], axis=0)
        out1 = tl.where(is_nonempty, acc1 / safe_sum1, 0.0)
        lse1 = tl.where(
            is_nonempty,
            max1 + tl.log2(safe_sum1),
            -float("inf"),
        )
        tl.store(out_base + HEAD_DIM, out1)
        tl.store(lse_base + 1, lse1)

        acc2 = tl.sum(v * weight2[:, None], axis=0)
        out2 = tl.where(is_nonempty, acc2 / safe_sum2, 0.0)
        lse2 = tl.where(
            is_nonempty,
            max2 + tl.log2(safe_sum2),
            -float("inf"),
        )
        tl.store(out_base + 2 * HEAD_DIM, out2)
        tl.store(lse_base + 2, lse2)

        acc3 = tl.sum(v * weight3[:, None], axis=0)
        out3 = tl.where(is_nonempty, acc3 / safe_sum3, 0.0)
        lse3 = tl.where(
            is_nonempty,
            max3 + tl.log2(safe_sum3),
            -float("inf"),
        )
        tl.store(out_base + 3 * HEAD_DIM, out3)
        tl.store(lse_base + 3, lse3)
    else:
        acc0 = tl.sum(v * weight0[:, None], axis=0)
        acc1 = tl.sum(v * weight1[:, None], axis=0)
        acc2 = tl.sum(v * weight2[:, None], axis=0)
        acc3 = tl.sum(v * weight3[:, None], axis=0)

        out0 = tl.where(is_nonempty, acc0 / safe_sum0, 0.0)
        out1 = tl.where(is_nonempty, acc1 / safe_sum1, 0.0)
        out2 = tl.where(is_nonempty, acc2 / safe_sum2, 0.0)
        out3 = tl.where(is_nonempty, acc3 / safe_sum3, 0.0)

        tl.store(out_base, out0)
        tl.store(out_base + HEAD_DIM, out1)
        tl.store(out_base + 2 * HEAD_DIM, out2)
        tl.store(out_base + 3 * HEAD_DIM, out3)

        lse0 = tl.where(
            is_nonempty,
            max0 + tl.log2(safe_sum0),
            -float("inf"),
        )
        lse1 = tl.where(
            is_nonempty,
            max1 + tl.log2(safe_sum1),
            -float("inf"),
        )
        lse2 = tl.where(
            is_nonempty,
            max2 + tl.log2(safe_sum2),
            -float("inf"),
        )
        lse3 = tl.where(
            is_nonempty,
            max3 + tl.log2(safe_sum3),
            -float("inf"),
        )

        tl.store(lse_base, lse0)
        tl.store(lse_base + 1, lse1)
        tl.store(lse_base + 2, lse2)
        tl.store(lse_base + 3, lse3)


@triton.jit
def _general_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    indptr_ptr,
    indices_ptr,
    out_ptr,
    lse_ptr,
    sm_scale_log2: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kp: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vp: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_indptr: tl.constexpr,
    stride_indices: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_index = pid // 8
    kv_head = pid % 8
    first_q_head = kv_head * 4
    offsets_d = tl.arange(0, HEAD_DIM)

    q_base = (
        q_ptr
        + batch_index * stride_qb
        + first_q_head * stride_qh
        + offsets_d * stride_qd
    )
    q0 = tl.load(q_base).to(tl.float32) * sm_scale_log2
    q1 = tl.load(q_base + stride_qh).to(tl.float32) * sm_scale_log2
    q2 = tl.load(q_base + 2 * stride_qh).to(tl.float32) * sm_scale_log2
    q3 = tl.load(q_base + 3 * stride_qh).to(tl.float32) * sm_scale_log2

    acc0 = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    acc1 = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    acc2 = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    acc3 = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    max0 = -float("inf")
    max1 = -float("inf")
    max2 = -float("inf")
    max3 = -float("inf")
    sum0 = 0.0
    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0

    page_start = tl.load(
        indptr_ptr + batch_index * stride_indptr
    ).to(tl.int32)
    page_end = tl.load(
        indptr_ptr + (batch_index + 1) * stride_indptr
    ).to(tl.int32)

    block_offsets = tl.arange(0, BLOCK_N)
    position = page_start

    while position < page_end:
        positions = position + block_offsets
        valid = positions < page_end

        page_ids = tl.load(
            indices_ptr + positions * stride_indices,
            mask=valid,
            other=0,
        ).to(tl.int32)

        k_offsets = (
            page_ids[:, None] * stride_kp
            + kv_head * stride_kh
            + offsets_d[None, :] * stride_kd
        )
        k = tl.load(
            k_ptr + k_offsets,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)

        score0 = tl.sum(k * q0[None, :], axis=1)
        score1 = tl.sum(k * q1[None, :], axis=1)
        score2 = tl.sum(k * q2[None, :], axis=1)
        score3 = tl.sum(k * q3[None, :], axis=1)

        score0 = tl.where(valid, score0, -float("inf"))
        score1 = tl.where(valid, score1, -float("inf"))
        score2 = tl.where(valid, score2, -float("inf"))
        score3 = tl.where(valid, score3, -float("inf"))

        block_max0 = tl.max(score0, axis=0)
        block_max1 = tl.max(score1, axis=0)
        block_max2 = tl.max(score2, axis=0)
        block_max3 = tl.max(score3, axis=0)

        new_max0 = tl.maximum(max0, block_max0)
        new_max1 = tl.maximum(max1, block_max1)
        new_max2 = tl.maximum(max2, block_max2)
        new_max3 = tl.maximum(max3, block_max3)

        old_scale0 = tl.exp2(max0 - new_max0)
        old_scale1 = tl.exp2(max1 - new_max1)
        old_scale2 = tl.exp2(max2 - new_max2)
        old_scale3 = tl.exp2(max3 - new_max3)

        weight0 = tl.exp2(score0 - new_max0)
        weight1 = tl.exp2(score1 - new_max1)
        weight2 = tl.exp2(score2 - new_max2)
        weight3 = tl.exp2(score3 - new_max3)

        v_offsets = (
            page_ids[:, None] * stride_vp
            + kv_head * stride_vh
            + offsets_d[None, :] * stride_vd
        )
        v = tl.load(
            v_ptr + v_offsets,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)

        acc0 = acc0 * old_scale0 + tl.sum(
            v * weight0[:, None], axis=0
        )
        acc1 = acc1 * old_scale1 + tl.sum(
            v * weight1[:, None], axis=0
        )
        acc2 = acc2 * old_scale2 + tl.sum(
            v * weight2[:, None], axis=0
        )
        acc3 = acc3 * old_scale3 + tl.sum(
            v * weight3[:, None], axis=0
        )

        sum0 = sum0 * old_scale0 + tl.sum(weight0, axis=0)
        sum1 = sum1 * old_scale1 + tl.sum(weight1, axis=0)
        sum2 = sum2 * old_scale2 + tl.sum(weight2, axis=0)
        sum3 = sum3 * old_scale3 + tl.sum(weight3, axis=0)

        max0 = new_max0
        max1 = new_max1
        max2 = new_max2
        max3 = new_max3
        position += BLOCK_N

    is_nonempty = page_start < page_end

    safe_sum0 = tl.where(is_nonempty, sum0, 1.0)
    safe_sum1 = tl.where(is_nonempty, sum1, 1.0)
    safe_sum2 = tl.where(is_nonempty, sum2, 1.0)
    safe_sum3 = tl.where(is_nonempty, sum3, 1.0)

    out0 = tl.where(is_nonempty, acc0 / safe_sum0, 0.0)
    out1 = tl.where(is_nonempty, acc1 / safe_sum1, 0.0)
    out2 = tl.where(is_nonempty, acc2 / safe_sum2, 0.0)
    out3 = tl.where(is_nonempty, acc3 / safe_sum3, 0.0)

    out_base = (
        out_ptr
        + batch_index * 4096
        + first_q_head * HEAD_DIM
        + offsets_d
    )
    tl.store(out_base, out0)
    tl.store(out_base + HEAD_DIM, out1)
    tl.store(out_base + 2 * HEAD_DIM, out2)
    tl.store(out_base + 3 * HEAD_DIM, out3)

    lse0 = tl.where(
        is_nonempty,
        max0 + tl.log2(safe_sum0),
        -float("inf"),
    )
    lse1 = tl.where(
        is_nonempty,
        max1 + tl.log2(safe_sum1),
        -float("inf"),
    )
    lse2 = tl.where(
        is_nonempty,
        max2 + tl.log2(safe_sum2),
        -float("inf"),
    )
    lse3 = tl.where(
        is_nonempty,
        max3 + tl.log2(safe_sum3),
        -float("inf"),
    )

    lse_base = lse_ptr + batch_index * 32 + first_q_head
    tl.store(lse_base, lse0)
    tl.store(lse_base + 1, lse1)
    tl.store(lse_base + 2, lse2)
    tl.store(lse_base + 3, lse3)


def _check_tensor(name, tensor, dtype, ndim):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.ndim != ndim:
        raise ValueError(
            f"{name} must have {ndim} dimensions, got {tensor.ndim}"
        )


def _move_to_cuda(tensor, device):
    if tensor.device == device:
        return tensor
    if tensor.device.type == "cpu":
        return tensor.cuda(device=device)
    return tensor.to(device=device)


@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    _check_tensor("q", q, torch.bfloat16, 3)
    _check_tensor("k_cache", k_cache, torch.bfloat16, 4)
    _check_tensor("v_cache", v_cache, torch.bfloat16, 4)
    _check_tensor("kv_indptr", kv_indptr, torch.int32, 1)
    _check_tensor("kv_indices", kv_indices, torch.int32, 1)

    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, cache_head_dim = k_cache.shape

    if num_qo_heads != 32:
        raise ValueError(f"num_qo_heads must be 32, got {num_qo_heads}")
    if num_kv_heads != 8:
        raise ValueError(f"num_kv_heads must be 8, got {num_kv_heads}")
    if head_dim != 128:
        raise ValueError(f"head_dim must be 128, got {head_dim}")
    if cache_head_dim != 128:
        raise ValueError(
            f"k_cache head dimension must be 128, got {cache_head_dim}"
        )
    if page_size != 1:
        raise ValueError(f"page_size must be 1, got {page_size}")
    if v_cache.shape != k_cache.shape:
        raise ValueError("v_cache must have the same shape as k_cache")
    if kv_indptr.shape[0] != batch_size + 1:
        raise ValueError(
            "len_indptr must equal batch_size + 1: "
            f"got {kv_indptr.shape[0]} and {batch_size + 1}"
        )

    maximum_kv_length = None
    fixed_kv_length = None

    if kv_indptr.device.type == "cpu":
        expected_indices = int(kv_indptr[-1].item())
        if expected_indices != kv_indices.shape[0]:
            raise ValueError(
                "num_kv_indices must equal kv_indptr[-1]: "
                f"got {kv_indices.shape[0]} and {expected_indices}"
            )
        if expected_indices < 0:
            raise ValueError("kv_indptr[-1] must be nonnegative")

        if batch_size > 0:
            lengths = kv_indptr[1:] - kv_indptr[:-1]
            if bool(torch.any(lengths < 0).item()):
                raise ValueError("kv_indptr must be nondecreasing")
            minimum_kv_length = int(lengths.min().item())
            maximum_kv_length = int(lengths.max().item())
            if minimum_kv_length == maximum_kv_length:
                fixed_kv_length = maximum_kv_length
        else:
            maximum_kv_length = 0
            fixed_kv_length = 0

    if isinstance(sm_scale, torch.Tensor):
        if sm_scale.numel() != 1:
            raise ValueError(
                "sm_scale tensor must contain exactly one element"
            )
        scale_value = float(sm_scale.detach().item())
    else:
        scale_value = float(sm_scale)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to execute the Triton GQA paged decode kernel"
        )

    original_output_device = q.device

    if q.device.type == "cuda":
        execution_device = q.device
    else:
        execution_device = None
        for tensor in (k_cache, v_cache, kv_indptr, kv_indices):
            if tensor.device.type == "cuda":
                execution_device = tensor.device
                break
        if execution_device is None:
            execution_device = torch.device(
                "cuda", torch.cuda.current_device()
            )

    q_gpu = _move_to_cuda(q, execution_device)
    k_gpu = _move_to_cuda(k_cache, execution_device)
    v_gpu = _move_to_cuda(v_cache, execution_device)
    indptr_gpu = _move_to_cuda(kv_indptr, execution_device)
    indices_gpu = _move_to_cuda(kv_indices, execution_device)

    output_gpu = torch.empty(
        (batch_size, 32, 128),
        dtype=torch.bfloat16,
        device=execution_device,
    )
    lse_gpu = torch.empty(
        (batch_size, 32),
        dtype=torch.float32,
        device=execution_device,
    )

    if batch_size > 0:
        scale_log2 = scale_value / math.log(2.0)
        average_kv_length = kv_indices.shape[0] / batch_size

        if maximum_kv_length is None and kv_indices.shape[0] <= 16:
            maximum_kv_length = kv_indices.shape[0]

        use_single_kernel = (
            maximum_kv_length is not None
            and maximum_kv_length <= 1
        )
        use_short_kernel = (
            maximum_kv_length is not None
            and maximum_kv_length <= 16
        )

        common_args = (
            q_gpu,
            k_gpu,
            v_gpu,
            indptr_gpu,
            indices_gpu,
            output_gpu,
            lse_gpu,
            scale_log2,
            q_gpu.stride(0),
            q_gpu.stride(1),
            q_gpu.stride(2),
            k_gpu.stride(0),
            k_gpu.stride(2),
            k_gpu.stride(3),
            v_gpu.stride(0),
            v_gpu.stride(2),
            v_gpu.stride(3),
            indptr_gpu.stride(0),
            indices_gpu.stride(0),
        )

        with torch.cuda.device(execution_device):
            if use_single_kernel:
                _single_kernel[(batch_size * 8,)](
                    *common_args,
                    HEAD_DIM=128,
                    num_warps=4,
                    num_stages=1,
                )
            elif use_short_kernel:
                compile_fixed_length = (
                    fixed_kv_length
                    if fixed_kv_length is not None
                    and fixed_kv_length > 0
                    else 0
                )

                if compile_fixed_length == 2:
                    block_n = 2
                    num_warps = 4
                    serial_accum = False
                elif compile_fixed_length <= 4 and compile_fixed_length > 0:
                    block_n = 4
                    num_warps = 4
                    serial_accum = False
                elif compile_fixed_length <= 8 and compile_fixed_length > 0:
                    block_n = 8
                    num_warps = 8
                    serial_accum = True
                elif compile_fixed_length <= 16 and compile_fixed_length > 0:
                    block_n = 16
                    num_warps = 8
                    serial_accum = True
                elif maximum_kv_length <= 2:
                    block_n = 2
                    num_warps = 4
                    serial_accum = False
                elif maximum_kv_length <= 4:
                    block_n = 4
                    num_warps = 4
                    serial_accum = False
                elif maximum_kv_length <= 8:
                    block_n = 8
                    num_warps = 4
                    serial_accum = False
                elif batch_size <= 16:
                    block_n = 16
                    num_warps = 8
                    serial_accum = False
                else:
                    block_n = 16
                    num_warps = 4
                    serial_accum = False

                _short_kernel[(batch_size * 8,)](
                    *common_args,
                    HEAD_DIM=128,
                    BLOCK_N=block_n,
                    FIXED_LENGTH=compile_fixed_length,
                    SERIAL_ACCUM=serial_accum,
                    num_warps=num_warps,
                    num_stages=1,
                )
            else:
                if average_kv_length <= 2:
                    block_n = 2
                    num_warps = 4
                elif average_kv_length <= 4:
                    block_n = 4
                    num_warps = 4
                elif average_kv_length <= 8:
                    block_n = 8
                    num_warps = 4
                elif batch_size <= 16:
                    block_n = 16
                    num_warps = 8
                else:
                    block_n = 16
                    num_warps = 4

                _general_kernel[(batch_size * 8,)](
                    *common_args,
                    HEAD_DIM=128,
                    BLOCK_N=block_n,
                    num_warps=num_warps,
                    num_stages=1,
                )

    if original_output_device.type == "cpu":
        return output_gpu.cpu(), lse_gpu.cpu()

    if original_output_device != execution_device:
        return (
            output_gpu.to(device=original_output_device),
            lse_gpu.to(device=original_output_device),
        )

    return output_gpu, lse_gpu