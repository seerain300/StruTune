# task: mla_paged_decode_h16_ckv512_kpe64_ps1
# bench: FlashInfer | batch: formal_20260914
# final eval (official evaluator, full workloads): valid=True pass=47/47 geomean=28.400x
# feedback best (5-workload sample during search): 36.789x
# torch fallback audit: 干净 (-)
# tokens: 1,935,702

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _singleton_attention_kernel(
    q_nope,
    q_pe,
    ckv_cache,
    kpe_cache,
    kv_indptr,
    kv_indices,
    output,
    lse,
    sm_scale,
):
    batch_idx = tl.program_id(0)

    sequence_begin = tl.load(kv_indptr + batch_idx)
    sequence_end = tl.load(kv_indptr + batch_idx + 1)
    has_token = sequence_end > sequence_begin

    offs_h = tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    offs_ckv = tl.arange(0, 512)
    offs_kpe = tl.arange(0, 64)

    token_mask = has_token & (offs_n == 0)
    pages = tl.load(
        kv_indices + sequence_begin + offs_n,
        mask=token_mask,
        other=0,
    ).to(tl.int64)

    qn = tl.load(
        q_nope
        + batch_idx * 16 * 512
        + offs_h[:, None] * 512
        + offs_ckv[None, :]
    )
    kc = tl.load(
        ckv_cache
        + pages[None, :] * 512
        + offs_ckv[:, None],
        mask=token_mask[None, :],
        other=0.0,
    )

    qp = tl.load(
        q_pe
        + batch_idx * 16 * 64
        + offs_h[:, None] * 64
        + offs_kpe[None, :]
    )
    kp = tl.load(
        kpe_cache
        + pages[None, :] * 64
        + offs_kpe[:, None],
        mask=token_mask[None, :],
        other=0.0,
    )

    logits = (tl.dot(qn, kc) + tl.dot(qp, kp)) * sm_scale
    singleton_logit = tl.sum(logits, axis=1)

    lse_value = tl.where(
        has_token,
        singleton_logit * 1.4426950408889634,
        -float("inf"),
    )
    tl.store(lse + batch_idx * 16 + offs_h, lse_value)

    page = tl.load(
        kv_indices + sequence_begin,
        mask=has_token,
        other=0,
    ).to(tl.int64)
    values = tl.load(
        ckv_cache + page * 512 + offs_ckv,
        mask=has_token,
        other=0.0,
    )

    output_offsets = (
        batch_idx * 16 * 512
        + offs_h[:, None] * 512
        + offs_ckv[None, :]
    )
    tl.store(
        output + output_offsets,
        values[None, :],
    )


@triton.jit
def _store_short_output_tile(
    probabilities,
    ckv_cache,
    output,
    pages,
    mask_n,
    batch_idx,
    offs_h,
    D_START: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    offs_d = D_START + tl.arange(0, BLOCK_D)

    values = tl.load(
        ckv_cache
        + pages[:, None] * 512
        + offs_d[None, :],
        mask=mask_n[:, None],
        other=0.0,
    )
    result = tl.dot(probabilities.to(tl.bfloat16), values)

    output_offsets = (
        batch_idx * 16 * 512
        + offs_h[:, None] * 512
        + offs_d[None, :]
    )
    tl.store(output + output_offsets, result)


@triton.jit
def _sequence_wide_short_kernel(
    q_nope,
    q_pe,
    ckv_cache,
    kpe_cache,
    kv_indptr,
    kv_indices,
    output,
    lse,
    sm_scale,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FILTER_LONG: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    sequence_begin = tl.load(kv_indptr + batch_idx)
    sequence_end = tl.load(kv_indptr + batch_idx + 1)
    sequence_length = sequence_end - sequence_begin

    if (not FILTER_LONG) or (sequence_length <= BLOCK_N):
        offs_h = tl.arange(0, 16)
        offs_n = tl.arange(0, BLOCK_N)
        offs_ckv = tl.arange(0, 512)
        offs_kpe = tl.arange(0, 64)
        mask_n = offs_n < sequence_length

        pages = tl.load(
            kv_indices + sequence_begin + offs_n,
            mask=mask_n,
            other=0,
        ).to(tl.int64)

        qn = tl.load(
            q_nope
            + batch_idx * 16 * 512
            + offs_h[:, None] * 512
            + offs_ckv[None, :]
        )
        kc = tl.load(
            ckv_cache
            + pages[None, :] * 512
            + offs_ckv[:, None],
            mask=mask_n[None, :],
            other=0.0,
        )

        qp = tl.load(
            q_pe
            + batch_idx * 16 * 64
            + offs_h[:, None] * 64
            + offs_kpe[None, :]
        )
        kp = tl.load(
            kpe_cache
            + pages[None, :] * 64
            + offs_kpe[:, None],
            mask=mask_n[None, :],
            other=0.0,
        )

        logits = (tl.dot(qn, kc) + tl.dot(qp, kp)) * sm_scale
        logits = tl.where(
            mask_n[None, :],
            logits,
            -float("inf"),
        )

        row_max = tl.max(logits, axis=1)
        has_tokens = sequence_length > 0
        safe_max = tl.where(has_tokens, row_max, 0.0)

        probabilities = tl.where(
            mask_n[None, :],
            tl.exp2(
                (logits - safe_max[:, None])
                * 1.4426950408889634
            ),
            0.0,
        )
        denominator = tl.sum(probabilities, axis=1)
        inv_denominator = tl.where(
            has_tokens,
            1.0 / denominator,
            0.0,
        )
        probabilities *= inv_denominator[:, None]

        lse_value = (
            safe_max * 1.4426950408889634
            + tl.log2(denominator)
        )
        lse_value = tl.where(
            has_tokens,
            lse_value,
            -float("inf"),
        )
        tl.store(
            lse + batch_idx * 16 + offs_h,
            lse_value,
        )

        _store_short_output_tile(
            probabilities,
            ckv_cache,
            output,
            pages,
            mask_n,
            batch_idx,
            offs_h,
            D_START=0,
            BLOCK_D=BLOCK_D,
        )
        _store_short_output_tile(
            probabilities,
            ckv_cache,
            output,
            pages,
            mask_n,
            batch_idx,
            offs_h,
            D_START=128,
            BLOCK_D=BLOCK_D,
        )
        _store_short_output_tile(
            probabilities,
            ckv_cache,
            output,
            pages,
            mask_n,
            batch_idx,
            offs_h,
            D_START=256,
            BLOCK_D=BLOCK_D,
        )
        _store_short_output_tile(
            probabilities,
            ckv_cache,
            output,
            pages,
            mask_n,
            batch_idx,
            offs_h,
            D_START=384,
            BLOCK_D=BLOCK_D,
        )


@triton.jit
def _head_group_attention_kernel(
    q_nope,
    q_pe,
    ckv_cache,
    kpe_cache,
    kv_indptr,
    kv_indices,
    output,
    lse,
    sm_scale,
    max_sequence_length,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_H: tl.constexpr,
    DYNAMIC_LENGTH: tl.constexpr,
    MIN_LENGTH: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_group_idx = tl.program_id(1)
    output_tile_idx = tl.program_id(2)

    sequence_begin = tl.load(kv_indptr + batch_idx)
    sequence_end = tl.load(kv_indptr + batch_idx + 1)
    sequence_length = sequence_end - sequence_begin

    if sequence_length > MIN_LENGTH:
        offs_h = (
            head_group_idx * GROUP_H
            + tl.arange(0, GROUP_H)
        )
        offs_n = tl.arange(0, BLOCK_N)
        offs_ckv = tl.arange(0, 512)
        offs_kpe = tl.arange(0, 64)
        offs_d = (
            output_tile_idx * BLOCK_D
            + tl.arange(0, BLOCK_D)
        )

        qn = tl.load(
            q_nope
            + batch_idx * 16 * 512
            + offs_h[:, None] * 512
            + offs_ckv[None, :]
        )
        qp = tl.load(
            q_pe
            + batch_idx * 16 * 64
            + offs_h[:, None] * 64
            + offs_kpe[None, :]
        )

        running_max = tl.full(
            (GROUP_H,),
            -float("inf"),
            tl.float32,
        )
        running_sum = tl.zeros(
            (GROUP_H,),
            tl.float32,
        )
        accumulator = tl.zeros(
            (GROUP_H, BLOCK_D),
            tl.float32,
        )

        loop_end = (
            sequence_length
            if DYNAMIC_LENGTH
            else max_sequence_length
        )

        for token_start in tl.range(
            0,
            loop_end,
            BLOCK_N,
            num_stages=1,
        ):
            token_offsets = token_start + offs_n
            token_mask = token_offsets < sequence_length

            pages = tl.load(
                kv_indices
                + sequence_begin
                + token_offsets,
                mask=token_mask,
                other=0,
            ).to(tl.int64)

            kc = tl.load(
                ckv_cache
                + pages[None, :] * 512
                + offs_ckv[:, None],
                mask=token_mask[None, :],
                other=0.0,
            )
            kp = tl.load(
                kpe_cache
                + pages[None, :] * 64
                + offs_kpe[:, None],
                mask=token_mask[None, :],
                other=0.0,
            )

            logits = (
                tl.dot(qn, kc)
                + tl.dot(qp, kp)
            ) * sm_scale
            logits = tl.where(
                token_mask[None, :],
                logits,
                -float("inf"),
            )

            block_max = tl.max(logits, axis=1)
            next_max = tl.maximum(
                running_max,
                block_max,
            )

            old_scale = tl.exp2(
                (running_max - next_max)
                * 1.4426950408889634
            )
            weights = tl.where(
                token_mask[None, :],
                tl.exp2(
                    (logits - next_max[:, None])
                    * 1.4426950408889634
                ),
                0.0,
            )

            values = tl.load(
                ckv_cache
                + pages[:, None] * 512
                + offs_d[None, :],
                mask=token_mask[:, None],
                other=0.0,
            )

            block_output = tl.dot(
                weights.to(tl.bfloat16),
                values,
            )
            accumulator = (
                accumulator * old_scale[:, None]
                + block_output
            )
            running_sum = (
                running_sum * old_scale
                + tl.sum(weights, axis=1)
            )
            running_max = next_max

        result = accumulator / running_sum[:, None]

        output_offsets = (
            batch_idx * 16 * 512
            + offs_h[:, None] * 512
            + offs_d[None, :]
        )
        tl.store(output + output_offsets, result)

        if output_tile_idx == 0:
            lse_value = (
                running_max * 1.4426950408889634
                + tl.log2(running_sum)
            )
            tl.store(
                lse + batch_idx * 16 + offs_h,
                lse_value,
            )


def _validate_inputs(
    q_nope,
    q_pe,
    ckv_cache,
    kpe_cache,
    kv_indptr,
    kv_indices,
):
    tensors = {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    if q_nope.ndim != 3 or q_nope.shape[1:] != (16, 512):
        raise ValueError(
            "q_nope must have shape [batch_size, 16, 512]"
        )
    if q_pe.shape != (q_nope.shape[0], 16, 64):
        raise ValueError(
            "q_pe must have shape [batch_size, 16, 64]"
        )
    if (
        ckv_cache.ndim != 3
        or ckv_cache.shape[1:] != (1, 512)
    ):
        raise ValueError(
            "ckv_cache must have shape [num_pages, 1, 512]"
        )
    if kpe_cache.shape != (ckv_cache.shape[0], 1, 64):
        raise ValueError(
            "kpe_cache must have shape [num_pages, 1, 64]"
        )
    if (
        kv_indptr.ndim != 1
        or kv_indptr.numel() != q_nope.shape[0] + 1
    ):
        raise ValueError(
            "kv_indptr length must equal batch_size + 1"
        )
    if kv_indices.ndim != 1:
        raise ValueError(
            "kv_indices must be one-dimensional"
        )

    if q_nope.dtype != torch.bfloat16:
        raise TypeError(
            "q_nope must have dtype torch.bfloat16"
        )
    if q_pe.dtype != torch.bfloat16:
        raise TypeError(
            "q_pe must have dtype torch.bfloat16"
        )
    if ckv_cache.dtype != torch.bfloat16:
        raise TypeError(
            "ckv_cache must have dtype torch.bfloat16"
        )
    if kpe_cache.dtype != torch.bfloat16:
        raise TypeError(
            "kpe_cache must have dtype torch.bfloat16"
        )
    if kv_indptr.dtype != torch.int32:
        raise TypeError(
            "kv_indptr must have dtype torch.int32"
        )
    if kv_indices.dtype != torch.int32:
        raise TypeError(
            "kv_indices must have dtype torch.int32"
        )


def _to_target_cuda(tensor, target_device):
    if tensor.device.type == "cpu":
        return tensor.cuda(
            device=target_device,
            non_blocking=False,
        )
    if tensor.device.type == "cuda":
        if tensor.device != target_device:
            return tensor.to(device=target_device)
        return tensor
    raise ValueError(
        f"unsupported tensor device: {tensor.device}"
    )


def _launch_long_kernel(
    batch_size,
    q_nope,
    q_pe,
    ckv_cache,
    kpe_cache,
    kv_indptr,
    kv_indices,
    output,
    lse,
    sm_scale,
    maximum_length,
    dynamic_length,
    minimum_length,
):
    if batch_size >= 4:
        group_h = 4
        head_groups = 4
        block_d = 128
        output_tiles = 4
    else:
        group_h = 2
        head_groups = 8
        block_d = 128
        output_tiles = 4

    _head_group_attention_kernel[
        (batch_size, head_groups, output_tiles)
    ](
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        kv_indptr,
        kv_indices,
        output,
        lse,
        sm_scale,
        maximum_length,
        BLOCK_N=32,
        BLOCK_D=block_d,
        GROUP_H=group_h,
        DYNAMIC_LENGTH=dynamic_length,
        MIN_LENGTH=minimum_length,
        num_warps=4,
        num_stages=1,
    )


@torch.no_grad()
def run(
    q_nope,
    q_pe,
    ckv_cache,
    kpe_cache,
    kv_indptr,
    kv_indices,
    sm_scale,
):
    _validate_inputs(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        kv_indptr,
        kv_indices,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to execute this Triton kernel"
        )

    original_device = q_nope.device
    if original_device.type == "cuda":
        target_device = original_device
    elif original_device.type == "cpu":
        target_device = torch.device(
            "cuda",
            torch.cuda.current_device(),
        )
    else:
        raise ValueError(
            f"unsupported q_nope device: {original_device}"
        )

    if isinstance(sm_scale, torch.Tensor):
        if sm_scale.numel() != 1:
            raise ValueError("sm_scale must be a scalar")
        sm_scale_value = float(sm_scale.detach().item())
    else:
        sm_scale_value = float(sm_scale)

    if not math.isfinite(sm_scale_value):
        raise ValueError("sm_scale must be finite")

    batch_size = q_nope.shape[0]
    total_indices = kv_indices.numel()
    host_indptr = None
    maximum_length = 0

    if kv_indptr.device.type == "cpu":
        host_indptr = kv_indptr.tolist()

        if host_indptr[-1] != total_indices:
            raise ValueError(
                "kv_indices length must equal kv_indptr[-1]"
            )

        for index in range(batch_size):
            length = (
                host_indptr[index + 1]
                - host_indptr[index]
            )
            if length < 0:
                raise ValueError(
                    "kv_indptr must be nondecreasing"
                )
            maximum_length = max(maximum_length, length)

    q_nope_gpu = _to_target_cuda(
        q_nope,
        target_device,
    ).contiguous()
    q_pe_gpu = _to_target_cuda(
        q_pe,
        target_device,
    ).contiguous()
    ckv_cache_gpu = _to_target_cuda(
        ckv_cache,
        target_device,
    ).contiguous()
    kpe_cache_gpu = _to_target_cuda(
        kpe_cache,
        target_device,
    ).contiguous()
    kv_indptr_gpu = _to_target_cuda(
        kv_indptr,
        target_device,
    ).contiguous()
    kv_indices_gpu = _to_target_cuda(
        kv_indices,
        target_device,
    ).contiguous()

    output_gpu = torch.empty_like(q_nope_gpu)
    lse_gpu = torch.empty(
        (batch_size, 16),
        dtype=torch.float32,
        device=target_device,
    )

    if batch_size == 0:
        if original_device.type == "cpu":
            return output_gpu.cpu(), lse_gpu.cpu()
        return output_gpu, lse_gpu

    with torch.cuda.device(target_device):
        if host_indptr is not None:
            if maximum_length <= 1:
                _singleton_attention_kernel[(batch_size,)](
                    q_nope_gpu,
                    q_pe_gpu,
                    ckv_cache_gpu,
                    kpe_cache_gpu,
                    kv_indptr_gpu,
                    kv_indices_gpu,
                    output_gpu,
                    lse_gpu,
                    sm_scale_value,
                    num_warps=8,
                    num_stages=2,
                )
            elif maximum_length <= 32:
                block_n = (
                    16 if maximum_length <= 16 else 32
                )
                _sequence_wide_short_kernel[(batch_size,)](
                    q_nope_gpu,
                    q_pe_gpu,
                    ckv_cache_gpu,
                    kpe_cache_gpu,
                    kv_indptr_gpu,
                    kv_indices_gpu,
                    output_gpu,
                    lse_gpu,
                    sm_scale_value,
                    BLOCK_N=block_n,
                    BLOCK_D=128,
                    FILTER_LONG=False,
                    num_warps=8,
                    num_stages=2,
                )
            else:
                _launch_long_kernel(
                    batch_size,
                    q_nope_gpu,
                    q_pe_gpu,
                    ckv_cache_gpu,
                    kpe_cache_gpu,
                    kv_indptr_gpu,
                    kv_indices_gpu,
                    output_gpu,
                    lse_gpu,
                    sm_scale_value,
                    maximum_length,
                    False,
                    0,
                )
        elif total_indices <= 32:
            block_n = 16 if total_indices <= 16 else 32
            _sequence_wide_short_kernel[(batch_size,)](
                q_nope_gpu,
                q_pe_gpu,
                ckv_cache_gpu,
                kpe_cache_gpu,
                kv_indptr_gpu,
                kv_indices_gpu,
                output_gpu,
                lse_gpu,
                sm_scale_value,
                BLOCK_N=block_n,
                BLOCK_D=128,
                FILTER_LONG=False,
                num_warps=8,
                num_stages=2,
            )
        else:
            _sequence_wide_short_kernel[(batch_size,)](
                q_nope_gpu,
                q_pe_gpu,
                ckv_cache_gpu,
                kpe_cache_gpu,
                kv_indptr_gpu,
                kv_indices_gpu,
                output_gpu,
                lse_gpu,
                sm_scale_value,
                BLOCK_N=32,
                BLOCK_D=128,
                FILTER_LONG=True,
                num_warps=8,
                num_stages=2,
            )
            _launch_long_kernel(
                batch_size,
                q_nope_gpu,
                q_pe_gpu,
                ckv_cache_gpu,
                kpe_cache_gpu,
                kv_indptr_gpu,
                kv_indices_gpu,
                output_gpu,
                lse_gpu,
                sm_scale_value,
                0,
                True,
                32,
            )

    if original_device.type == "cpu":
        return output_gpu.cpu(), lse_gpu.cpu()

    return output_gpu, lse_gpu