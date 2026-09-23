# solution=GPT-5.6-Sol_gqa_ragged_prefill_causal_h32_kv8_d128_triton_optimized_r11 score=7.748460866152776 passed=False
import weakref

import torch
import triton
import triton.language as tl


@triton.jit
def _single_token_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    lse_ptr,
    sm_scale,
    stride_qh,
    stride_qd,
    stride_kh,
    stride_kd,
    stride_vh,
    stride_vd,
    HEAD_DIM: tl.constexpr,
    GQA_RATIO: tl.constexpr,
):
    kv_head = tl.program_id(0)
    d = tl.arange(0, HEAD_DIM)
    qh = kv_head * GQA_RATIO

    k = tl.load(k_ptr + kv_head * stride_kh + d * stride_kd).to(tl.float32)
    v = tl.load(v_ptr + kv_head * stride_vh + d * stride_vd)

    qbase = qh * stride_qh + d * stride_qd
    q0 = tl.load(q_ptr + qbase).to(tl.float32)
    q1 = tl.load(q_ptr + qbase + stride_qh).to(tl.float32)
    q2 = tl.load(q_ptr + qbase + 2 * stride_qh).to(tl.float32)
    q3 = tl.load(q_ptr + qbase + 3 * stride_qh).to(tl.float32)

    log2_scale = sm_scale * 1.4426950408889634
    l0 = tl.sum(q0 * k, axis=0) * log2_scale
    l1 = tl.sum(q1 * k, axis=0) * log2_scale
    l2 = tl.sum(q2 * k, axis=0) * log2_scale
    l3 = tl.sum(q3 * k, axis=0) * log2_scale

    obase = qh * HEAD_DIM + d
    tl.store(output_ptr + obase, v)
    tl.store(output_ptr + obase + HEAD_DIM, v)
    tl.store(output_ptr + obase + 2 * HEAD_DIM, v)
    tl.store(output_ptr + obase + 3 * HEAD_DIM, v)

    tl.store(lse_ptr + qh, l0)
    tl.store(lse_ptr + qh + 1, l1)
    tl.store(lse_ptr + qh + 2, l2)
    tl.store(lse_ptr + qh + 3, l3)


@triton.jit
def _sequence_tiled_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    output_ptr,
    lse_ptr,
    sm_scale,
    total_q,
    total_kv,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_indptr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    SINGLE_SEQUENCE: tl.constexpr,
):
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    tile = tl.program_id(2)

    if SINGLE_SEQUENCE:
        q_start = 0
        kv_start = 0
        q_len = total_q
        kv_len = total_kv
    else:
        q_start = tl.load(qo_indptr_ptr + batch * stride_indptr)
        q_end = tl.load(qo_indptr_ptr + (batch + 1) * stride_indptr)
        kv_start = tl.load(kv_indptr_ptr + batch * stride_indptr)
        kv_end = tl.load(kv_indptr_ptr + (batch + 1) * stride_indptr)
        q_len = q_end - q_start
        kv_len = kv_end - kv_start

    tile_start = tile * BLOCK_Q
    if tile_start >= q_len:
        return

    rows = tl.arange(0, BLOCK_M)
    d = tl.arange(0, HEAD_DIM)

    qpos = tile_start + rows // GQA_RATIO
    qhead = kv_head * GQA_RATIO + rows % GQA_RATIO
    qmask = qpos < q_len
    global_q = q_start + qpos

    q_offsets = (
        global_q[:, None] * stride_qt
        + qhead[:, None] * stride_qh
        + d[None, :] * stride_qd
    )
    q = tl.load(q_ptr + q_offsets, mask=qmask[:, None], other=0.0)

    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    norm = tl.zeros((BLOCK_M,), dtype=tl.float32)
    rmax = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)

    log2_scale = sm_scale * 1.4426950408889634
    delta = kv_len - q_len

    kv_limit = tl.minimum(kv_len, tile_start + BLOCK_Q + delta)
    kv_limit = tl.maximum(kv_limit, 0)

    full_limit = tl.minimum(
        kv_len,
        tl.maximum(tile_start + 1 + delta, 0),
    )
    full_limit = (full_limit // BLOCK_N) * BLOCK_N

    for start in range(0, full_limit, BLOCK_N):
        ko = start + tl.arange(0, BLOCK_N)
        gk = kv_start + ko

        koff = (
            gk[None, :] * stride_kt
            + kv_head * stride_kh
            + d[:, None] * stride_kd
        )
        k = tl.load(k_ptr + koff)

        logits = tl.dot(q, k) * log2_scale
        block_max = tl.max(logits, axis=1)
        new_max = tl.maximum(rmax, block_max)
        correction = tl.exp2(rmax - new_max)
        probs = tl.exp2(logits - new_max[:, None])

        voff = (
            gk[:, None] * stride_vt
            + kv_head * stride_vh
            + d[None, :] * stride_vd
        )
        v = tl.load(v_ptr + voff)

        acc = acc * correction[:, None] + tl.dot(
            probs.to(tl.bfloat16),
            v,
        )
        norm = norm * correction + tl.sum(probs, axis=1)
        rmax = new_max

    for start in range(full_limit, kv_limit, BLOCK_N):
        ko = start + tl.arange(0, BLOCK_N)
        kmask = ko < kv_limit
        gk = kv_start + ko

        koff = (
            gk[None, :] * stride_kt
            + kv_head * stride_kh
            + d[:, None] * stride_kd
        )
        k = tl.load(
            k_ptr + koff,
            mask=kmask[None, :],
            other=0.0,
        )

        logits = tl.dot(q, k) * log2_scale
        causal = ko[None, :] < (qpos[:, None] + 1 + delta)
        mask = qmask[:, None] & kmask[None, :] & causal
        logits = tl.where(mask, logits, -float("inf"))

        block_max = tl.max(logits, axis=1)
        new_max = tl.maximum(rmax, block_max)
        finite = new_max != -float("inf")

        correction = tl.where(
            finite,
            tl.exp2(rmax - new_max),
            0.0,
        )
        probs = tl.where(
            mask,
            tl.exp2(logits - new_max[:, None]),
            0.0,
        )

        voff = (
            gk[:, None] * stride_vt
            + kv_head * stride_vh
            + d[None, :] * stride_vd
        )
        v = tl.load(
            v_ptr + voff,
            mask=kmask[:, None],
            other=0.0,
        )

        acc = acc * correction[:, None] + tl.dot(
            probs.to(tl.bfloat16),
            v,
        )
        norm = norm * correction + tl.sum(probs, axis=1)
        rmax = new_max

    valid = norm > 0.0
    safe_norm = tl.where(valid, norm, 1.0)
    out = acc / safe_norm[:, None]
    out = tl.where(valid[:, None], out, 0.0)

    out_offsets = (
        global_q[:, None] * (32 * HEAD_DIM)
        + qhead[:, None] * HEAD_DIM
        + d[None, :]
    )
    tl.store(
        output_ptr + out_offsets,
        out,
        mask=qmask[:, None],
    )

    lse = tl.where(
        valid,
        rmax + tl.log2(safe_norm),
        -float("inf"),
    )
    tl.store(
        lse_ptr + global_q * 32 + qhead,
        lse,
        mask=qmask,
    )


@triton.jit
def _singleton_query_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    output_ptr,
    lse_ptr,
    sm_scale,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_indptr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GQA_RATIO: tl.constexpr,
):
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)

    qs = tl.load(qo_indptr_ptr + batch * stride_indptr)
    qe = tl.load(qo_indptr_ptr + (batch + 1) * stride_indptr)
    if qs >= qe:
        return

    ks = tl.load(kv_indptr_ptr + batch * stride_indptr)
    ke = tl.load(kv_indptr_ptr + (batch + 1) * stride_indptr)
    kv_len = ke - ks

    d = tl.arange(0, HEAD_DIM)
    qh = kv_head * GQA_RATIO
    qbase = qs * stride_qt + qh * stride_qh + d * stride_qd

    q0 = tl.load(q_ptr + qbase).to(tl.float32)
    q1 = tl.load(q_ptr + qbase + stride_qh).to(tl.float32)
    q2 = tl.load(q_ptr + qbase + 2 * stride_qh).to(tl.float32)
    q3 = tl.load(q_ptr + qbase + 3 * stride_qh).to(tl.float32)

    a0 = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    a1 = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    a2 = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    a3 = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    n0 = 0.0
    n1 = 0.0
    n2 = 0.0
    n3 = 0.0

    m0 = -float("inf")
    m1 = -float("inf")
    m2 = -float("inf")
    m3 = -float("inf")

    log2_scale = sm_scale * 1.4426950408889634

    for start in range(0, kv_len, BLOCK_N):
        ko = start + tl.arange(0, BLOCK_N)
        mask = ko < kv_len
        gk = ks + ko

        koff = (
            gk[:, None] * stride_kt
            + kv_head * stride_kh
            + d[None, :] * stride_kd
        )
        k = tl.load(
            k_ptr + koff,
            mask=mask[:, None],
            other=0.0,
        ).to(tl.float32)

        x0 = tl.where(
            mask,
            tl.sum(k * q0[None, :], axis=1) * log2_scale,
            -float("inf"),
        )
        x1 = tl.where(
            mask,
            tl.sum(k * q1[None, :], axis=1) * log2_scale,
            -float("inf"),
        )
        x2 = tl.where(
            mask,
            tl.sum(k * q2[None, :], axis=1) * log2_scale,
            -float("inf"),
        )
        x3 = tl.where(
            mask,
            tl.sum(k * q3[None, :], axis=1) * log2_scale,
            -float("inf"),
        )

        nm0 = tl.maximum(m0, tl.max(x0, axis=0))
        nm1 = tl.maximum(m1, tl.max(x1, axis=0))
        nm2 = tl.maximum(m2, tl.max(x2, axis=0))
        nm3 = tl.maximum(m3, tl.max(x3, axis=0))

        c0 = tl.exp2(m0 - nm0)
        c1 = tl.exp2(m1 - nm1)
        c2 = tl.exp2(m2 - nm2)
        c3 = tl.exp2(m3 - nm3)

        p0 = tl.where(mask, tl.exp2(x0 - nm0), 0.0)
        p1 = tl.where(mask, tl.exp2(x1 - nm1), 0.0)
        p2 = tl.where(mask, tl.exp2(x2 - nm2), 0.0)
        p3 = tl.where(mask, tl.exp2(x3 - nm3), 0.0)

        voff = (
            gk[:, None] * stride_vt
            + kv_head * stride_vh
            + d[None, :] * stride_vd
        )
        v = tl.load(
            v_ptr + voff,
            mask=mask[:, None],
            other=0.0,
        ).to(tl.float32)

        a0 = a0 * c0 + tl.sum(p0[:, None] * v, axis=0)
        a1 = a1 * c1 + tl.sum(p1[:, None] * v, axis=0)
        a2 = a2 * c2 + tl.sum(p2[:, None] * v, axis=0)
        a3 = a3 * c3 + tl.sum(p3[:, None] * v, axis=0)

        n0 = n0 * c0 + tl.sum(p0, axis=0)
        n1 = n1 * c1 + tl.sum(p1, axis=0)
        n2 = n2 * c2 + tl.sum(p2, axis=0)
        n3 = n3 * c3 + tl.sum(p3, axis=0)

        m0 = nm0
        m1 = nm1
        m2 = nm2
        m3 = nm3

    h0 = n0 > 0.0
    h1 = n1 > 0.0
    h2 = n2 > 0.0
    h3 = n3 > 0.0

    s0 = tl.where(h0, n0, 1.0)
    s1 = tl.where(h1, n1, 1.0)
    s2 = tl.where(h2, n2, 1.0)
    s3 = tl.where(h3, n3, 1.0)

    obase = qs * 32 * HEAD_DIM + qh * HEAD_DIM + d
    tl.store(
        output_ptr + obase,
        tl.where(h0, a0 / s0, 0.0),
    )
    tl.store(
        output_ptr + obase + HEAD_DIM,
        tl.where(h1, a1 / s1, 0.0),
    )
    tl.store(
        output_ptr + obase + 2 * HEAD_DIM,
        tl.where(h2, a2 / s2, 0.0),
    )
    tl.store(
        output_ptr + obase + 3 * HEAD_DIM,
        tl.where(h3, a3 / s3, 0.0),
    )

    lbase = qs * 32 + qh
    tl.store(
        lse_ptr + lbase,
        tl.where(h0, m0 + tl.log2(s0), -float("inf")),
    )
    tl.store(
        lse_ptr + lbase + 1,
        tl.where(h1, m1 + tl.log2(s1), -float("inf")),
    )
    tl.store(
        lse_ptr + lbase + 2,
        tl.where(h2, m2 + tl.log2(s2), -float("inf")),
    )
    tl.store(
        lse_ptr + lbase + 3,
        tl.where(h3, m3 + tl.log2(s3), -float("inf")),
    )


_MAX_Q_LENGTH_CACHE = {}


def _cached_max_q_length(indptr):
    key = id(indptr)
    version = getattr(indptr, "_version", None)
    cached = _MAX_Q_LENGTH_CACHE.get(key)

    if cached is not None:
        ref, old_version, value = cached
        if ref() is indptr and old_version == version:
            return value
        _MAX_Q_LENGTH_CACHE.pop(key, None)

    value = int((indptr[1:] - indptr[:-1]).max().item())

    if len(_MAX_Q_LENGTH_CACHE) >= 64:
        dead = [
            cache_key
            for cache_key, entry in _MAX_Q_LENGTH_CACHE.items()
            if entry[0]() is None
        ]
        for cache_key in dead:
            _MAX_Q_LENGTH_CACHE.pop(cache_key, None)
        if len(_MAX_Q_LENGTH_CACHE) >= 64:
            _MAX_Q_LENGTH_CACHE.clear()

    _MAX_Q_LENGTH_CACHE[key] = (
        weakref.ref(indptr),
        version,
        value,
    )
    return value


def _select_cuda_device(tensors):
    for tensor in tensors:
        if tensor.is_cuda:
            return tensor.device
    return torch.device("cuda", torch.cuda.current_device())


def _move_to_cuda(tensor, device):
    if tensor.is_cuda and tensor.device == device:
        return tensor
    return tensor.cuda(device=device)


@torch.no_grad()
def run(
    q,
    k,
    v,
    qo_indptr,
    kv_indptr,
    sm_scale,
):
    tensors = (q, k, v, qo_indptr, kv_indptr)
    names = ("q", "k", "v", "qo_indptr", "kv_indptr")

    for tensor, name in zip(tensors, names):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    if not torch.cuda.is_available():
        if any(tensor.is_cuda for tensor in tensors):
            raise RuntimeError(
                "CUDA tensors were provided, but CUDA is not available"
            )
        raise RuntimeError(
            "gqa_ragged attention requires a CUDA-capable GPU"
        )

    if q.ndim != 3 or q.shape[1] != 32 or q.shape[2] != 128:
        raise ValueError("q must have shape [total_q, 32, 128]")
    if k.ndim != 3 or k.shape[1] != 8 or k.shape[2] != 128:
        raise ValueError("k must have shape [total_kv, 8, 128]")
    if v.shape != k.shape:
        raise ValueError("v must have the same shape as k")
    if qo_indptr.ndim != 1 or kv_indptr.ndim != 1:
        raise ValueError(
            "qo_indptr and kv_indptr must be one-dimensional"
        )
    if qo_indptr.shape != kv_indptr.shape or qo_indptr.numel() < 2:
        raise ValueError(
            "qo_indptr and kv_indptr must have the same length "
            "of at least 2"
        )
    if q.dtype != torch.bfloat16:
        raise TypeError("q must have dtype torch.bfloat16")
    if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("k and v must have dtype torch.bfloat16")
    if qo_indptr.dtype != torch.int32 or kv_indptr.dtype != torch.int32:
        raise TypeError(
            "qo_indptr and kv_indptr must have dtype torch.int32"
        )

    original_device = q.device
    target_device = _select_cuda_device(tensors)

    batch_size = qo_indptr.numel() - 1
    total_q = q.shape[0]
    total_kv = k.shape[0]

    if total_q == 0:
        max_q_length = 0
    elif batch_size == 1:
        max_q_length = total_q
    elif qo_indptr.is_cuda:
        max_q_length = None
    else:
        max_q_length = _cached_max_q_length(qo_indptr)

    if (
        q.is_cuda
        and q.device == target_device
        and k.is_cuda
        and k.device == target_device
        and v.is_cuda
        and v.device == target_device
        and qo_indptr.is_cuda
        and qo_indptr.device == target_device
        and kv_indptr.is_cuda
        and kv_indptr.device == target_device
    ):
        q_gpu = q
        k_gpu = k
        v_gpu = v
        qo_gpu = qo_indptr
        kv_gpu = kv_indptr
    else:
        q_gpu = _move_to_cuda(q, target_device)
        k_gpu = _move_to_cuda(k, target_device)
        v_gpu = _move_to_cuda(v, target_device)
        qo_gpu = _move_to_cuda(qo_indptr, target_device)
        kv_gpu = _move_to_cuda(kv_indptr, target_device)

    if isinstance(sm_scale, torch.Tensor):
        if sm_scale.numel() != 1:
            raise ValueError("sm_scale must be a scalar")
        scale = float(sm_scale.detach().item())
    else:
        scale = float(sm_scale)

    if max_q_length is None:
        max_q_length = _cached_max_q_length(qo_gpu)

    output = torch.empty_like(q_gpu)
    lse = torch.empty(
        (total_q, 32),
        dtype=torch.float32,
        device=target_device,
    )

    if total_q > 0 and max_q_length > 0:
        with torch.cuda.device(target_device):
            if batch_size == 1 and total_q == 1 and total_kv == 1:
                _single_token_attention_kernel[(8,)](
                    q_gpu,
                    k_gpu,
                    v_gpu,
                    output,
                    lse,
                    scale,
                    q_gpu.stride(1),
                    q_gpu.stride(2),
                    k_gpu.stride(1),
                    k_gpu.stride(2),
                    v_gpu.stride(1),
                    v_gpu.stride(2),
                    HEAD_DIM=128,
                    GQA_RATIO=4,
                    num_warps=4,
                    num_stages=1,
                )
            elif max_q_length == 1:
                _singleton_query_attention_kernel[
                    (batch_size, 8)
                ](
                    q_gpu,
                    k_gpu,
                    v_gpu,
                    qo_gpu,
                    kv_gpu,
                    output,
                    lse,
                    scale,
                    q_gpu.stride(0),
                    q_gpu.stride(1),
                    q_gpu.stride(2),
                    k_gpu.stride(0),
                    k_gpu.stride(1),
                    k_gpu.stride(2),
                    v_gpu.stride(0),
                    v_gpu.stride(1),
                    v_gpu.stride(2),
                    qo_gpu.stride(0),
                    BLOCK_N=16,
                    HEAD_DIM=128,
                    GQA_RATIO=4,
                    num_warps=4,
                    num_stages=2,
                )
            else:
                if max_q_length <= 32:
                    block_q = 8
                    block_n = 16
                    warps = 4
                    stages = 2
                elif max_q_length <= 64:
                    block_q = 16
                    block_n = 32
                    warps = 8
                    stages = 2
                elif max_q_length <= 256:
                    block_q = 16
                    block_n = 64
                    warps = 8
                    stages = 4
                else:
                    block_q = 32
                    block_n = 128
                    warps = 8
                    stages = 4

                grid = (
                    batch_size,
                    8,
                    triton.cdiv(max_q_length, block_q),
                )

                _sequence_tiled_attention_kernel[grid](
                    q_gpu,
                    k_gpu,
                    v_gpu,
                    qo_gpu,
                    kv_gpu,
                    output,
                    lse,
                    scale,
                    total_q,
                    total_kv,
                    q_gpu.stride(0),
                    q_gpu.stride(1),
                    q_gpu.stride(2),
                    k_gpu.stride(0),
                    k_gpu.stride(1),
                    k_gpu.stride(2),
                    v_gpu.stride(0),
                    v_gpu.stride(1),
                    v_gpu.stride(2),
                    qo_gpu.stride(0),
                    BLOCK_Q=block_q,
                    BLOCK_M=block_q * 4,
                    BLOCK_N=block_n,
                    HEAD_DIM=128,
                    GQA_RATIO=4,
                    SINGLE_SEQUENCE=batch_size == 1,
                    num_warps=warps,
                    num_stages=stages,
                )
    else:
        output.zero_()
        lse.fill_(-float("inf"))

    if original_device.type == "cuda" and original_device == target_device:
        return output, lse

    return output.to(original_device), lse.to(original_device)