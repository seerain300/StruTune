# solution=GPT-5.6-Sol_gdn_prefill_qk4_v8_d128_k_last_triton_optimized_r8 score=248.24841753742044 passed=True
import math

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_V": 1}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_V": 1}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_V": 2}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_V": 2}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_V": 4}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_V": 4}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_V": 8}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_V": 8}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_V": 16}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_V": 16}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_V": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_V": 32}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_V": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_V": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_V": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_V": 128}, num_warps=8, num_stages=1),
    ],
    key=["SEQ_LEN_BUCKET", "NUM_SEQS_BUCKET", "HAS_STATE"],
)
@triton.jit
def _gdn_prefill_paired_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    state_ptr,
    A_log_ptr,
    a_ptr,
    dt_bias_ptr,
    b_ptr,
    cu_seqlens_ptr,
    output_ptr,
    new_state_ptr,
    scale,
    SEQ_LEN_BUCKET,
    NUM_SEQS_BUCKET,
    HEAD_SIZE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    HAS_STATE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    pair_idx = tl.program_id(1)
    value_block = tl.program_id(2)

    value_head0 = pair_idx * 2
    value_head1 = value_head0 + 1
    query_head = pair_idx
    key_head = pair_idx

    value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    key_offsets = tl.arange(0, HEAD_SIZE)

    seq_start = tl.load(cu_seqlens_ptr + seq_idx).to(tl.int32)
    seq_end = tl.load(cu_seqlens_ptr + seq_idx + 1).to(tl.int32)

    state_offsets0 = (
        (
            (seq_idx * NUM_V_HEADS + value_head0) * HEAD_SIZE
            + value_offsets[:, None]
        )
        * HEAD_SIZE
        + key_offsets[None, :]
    )
    state_offsets1 = (
        (
            (seq_idx * NUM_V_HEADS + value_head1) * HEAD_SIZE
            + value_offsets[:, None]
        )
        * HEAD_SIZE
        + key_offsets[None, :]
    )

    state_values0 = tl.zeros((BLOCK_V, HEAD_SIZE), dtype=tl.float32)
    state_values1 = tl.zeros((BLOCK_V, HEAD_SIZE), dtype=tl.float32)

    if HAS_STATE:
        if seq_start < seq_end:
            state_values0 = tl.load(
                state_ptr + state_offsets0,
                cache_modifier=".cg",
            ).to(tl.float32)
            state_values1 = tl.load(
                state_ptr + state_offsets1,
                cache_modifier=".cg",
            ).to(tl.float32)

    decay_rate0 = tl.exp(
        tl.load(A_log_ptr + value_head0).to(tl.float32)
    )
    decay_rate1 = tl.exp(
        tl.load(A_log_ptr + value_head1).to(tl.float32)
    )
    decay_bias0 = tl.load(
        dt_bias_ptr + value_head0
    ).to(tl.float32)
    decay_bias1 = tl.load(
        dt_bias_ptr + value_head1
    ).to(tl.float32)

    token_idx = seq_start
    while token_idx < seq_end:
        key_base = (
            (token_idx * NUM_K_HEADS + key_head) * HEAD_SIZE
        )
        query_base = (
            (token_idx * NUM_Q_HEADS + query_head) * HEAD_SIZE
        )
        value_base0 = (
            (token_idx * NUM_V_HEADS + value_head0) * HEAD_SIZE
        )
        value_base1 = (
            (token_idx * NUM_V_HEADS + value_head1) * HEAD_SIZE
        )

        key_values = tl.load(
            k_ptr + key_base + key_offsets
        ).to(tl.float32)
        query_values = tl.load(
            q_ptr + query_base + key_offsets
        ).to(tl.float32)
        input_values0 = tl.load(
            v_ptr + value_base0 + value_offsets
        ).to(tl.float32)
        input_values1 = tl.load(
            v_ptr + value_base1 + value_offsets
        ).to(tl.float32)

        gate_offset0 = token_idx * NUM_V_HEADS + value_head0
        gate_offset1 = token_idx * NUM_V_HEADS + value_head1

        decay_input0 = (
            tl.load(a_ptr + gate_offset0).to(tl.float32)
            + decay_bias0
        )
        decay_input1 = (
            tl.load(a_ptr + gate_offset1).to(tl.float32)
            + decay_bias1
        )

        softplus0 = (
            tl.maximum(decay_input0, 0.0)
            + tl.log(1.0 + tl.exp(-tl.abs(decay_input0)))
        )
        softplus1 = (
            tl.maximum(decay_input1, 0.0)
            + tl.log(1.0 + tl.exp(-tl.abs(decay_input1)))
        )

        decay0 = tl.exp(-decay_rate0 * softplus0)
        decay1 = tl.exp(-decay_rate1 * softplus1)

        beta0 = tl.sigmoid(
            tl.load(b_ptr + gate_offset0).to(tl.float32)
        )
        beta1 = tl.sigmoid(
            tl.load(b_ptr + gate_offset1).to(tl.float32)
        )

        state_values0 *= decay0
        state_values1 *= decay1

        predicted_values0 = tl.sum(
            state_values0 * key_values[None, :],
            axis=1,
        )
        predicted_values1 = tl.sum(
            state_values1 * key_values[None, :],
            axis=1,
        )

        correction0 = beta0 * (input_values0 - predicted_values0)
        correction1 = beta1 * (input_values1 - predicted_values1)

        state_values0 += correction0[:, None] * key_values[None, :]
        state_values1 += correction1[:, None] * key_values[None, :]

        output_values0 = tl.sum(
            state_values0 * query_values[None, :],
            axis=1,
        ) * scale
        output_values1 = tl.sum(
            state_values1 * query_values[None, :],
            axis=1,
        ) * scale

        output_base0 = (
            (token_idx * NUM_V_HEADS + value_head0) * HEAD_SIZE
        )
        output_base1 = (
            (token_idx * NUM_V_HEADS + value_head1) * HEAD_SIZE
        )

        tl.store(
            output_ptr + output_base0 + value_offsets,
            output_values0,
        )
        tl.store(
            output_ptr + output_base1 + value_offsets,
            output_values1,
        )

        token_idx += 1

    tl.store(
        new_state_ptr + state_offsets0,
        state_values0,
        cache_modifier=".cs",
    )
    tl.store(
        new_state_ptr + state_offsets1,
        state_values1,
        cache_modifier=".cs",
    )


def _validate_tensor(name, tensor, dtype, ndim):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise TypeError(
            f"{name} must have dtype {dtype}, got {tensor.dtype}"
        )
    if tensor.ndim != ndim:
        raise ValueError(
            f"{name} must have {ndim} dimensions, got {tensor.ndim}"
        )


def _next_power_of_two(value):
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def run(
    q,
    k,
    v,
    state,
    A_log,
    a,
    dt_bias,
    b,
    cu_seqlens,
    scale,
):
    _validate_tensor("q", q, torch.bfloat16, 3)
    _validate_tensor("k", k, torch.bfloat16, 3)
    _validate_tensor("v", v, torch.bfloat16, 3)
    _validate_tensor("A_log", A_log, torch.float32, 1)
    _validate_tensor("a", a, torch.bfloat16, 2)
    _validate_tensor("dt_bias", dt_bias, torch.float32, 1)
    _validate_tensor("b", b, torch.bfloat16, 2)
    _validate_tensor("cu_seqlens", cu_seqlens, torch.int64, 1)

    if state is not None:
        _validate_tensor("state", state, torch.float32, 4)

    total_seq_len = q.shape[0]
    num_seqs = cu_seqlens.numel() - 1

    if num_seqs < 0:
        raise ValueError(
            "cu_seqlens must contain at least one element"
        )

    if q.shape != (total_seq_len, 4, 128):
        raise ValueError(
            f"q must have shape [{total_seq_len}, 4, 128]"
        )
    if k.shape != (total_seq_len, 4, 128):
        raise ValueError(
            f"k must have shape [{total_seq_len}, 4, 128]"
        )
    if v.shape != (total_seq_len, 8, 128):
        raise ValueError(
            f"v must have shape [{total_seq_len}, 8, 128]"
        )
    if a.shape != (total_seq_len, 8):
        raise ValueError(
            f"a must have shape [{total_seq_len}, 8]"
        )
    if b.shape != (total_seq_len, 8):
        raise ValueError(
            f"b must have shape [{total_seq_len}, 8]"
        )
    if A_log.shape != (8,):
        raise ValueError("A_log must have shape [8]")
    if dt_bias.shape != (8,):
        raise ValueError("dt_bias must have shape [8]")
    if state is not None and state.shape != (
        num_seqs,
        8,
        128,
        128,
    ):
        raise ValueError(
            f"state must have shape [{num_seqs}, 8, 128, 128]"
        )

    if not torch.cuda.is_available():
        cuda_inputs = [
            tensor
            for tensor in (
                q,
                k,
                v,
                state,
                A_log,
                a,
                dt_bias,
                b,
                cu_seqlens,
            )
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda
        ]
        if cuda_inputs:
            raise RuntimeError(
                "CUDA tensors were provided, but CUDA is not available"
            )
        raise RuntimeError(
            "gdn_prefill_qk4_v8_d128_k_last requires CUDA, "
            "but CUDA is not available"
        )

    tensors = (
        q,
        k,
        v,
        state,
        A_log,
        a,
        dt_bias,
        b,
        cu_seqlens,
    )
    cuda_devices = {
        tensor.device
        for tensor in tensors
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda
    }

    if len(cuda_devices) > 1:
        raise ValueError(
            "all CUDA input tensors must be on the same device"
        )

    if cuda_devices:
        target_device = next(iter(cuda_devices))
    else:
        target_device = torch.device(
            "cuda",
            torch.cuda.current_device(),
        )

    output_device = q.device
    state_output_device = (
        state.device if state is not None else q.device
    )

    def to_cuda(tensor):
        if tensor is None:
            return None
        if tensor.device == target_device:
            return tensor.contiguous()
        return tensor.cuda(device=target_device).contiguous()

    q_gpu = to_cuda(q)
    k_gpu = to_cuda(k)
    v_gpu = to_cuda(v)
    state_gpu = to_cuda(state)
    A_log_gpu = to_cuda(A_log)
    a_gpu = to_cuda(a)
    dt_bias_gpu = to_cuda(dt_bias)
    b_gpu = to_cuda(b)
    cu_seqlens_gpu = to_cuda(cu_seqlens)

    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            raise ValueError("scale must be a scalar")
        scale_value = float(scale.detach().item())
    elif scale is None:
        scale_value = 0.0
    else:
        scale_value = float(scale)

    if scale_value == 0.0:
        scale_value = 1.0 / math.sqrt(128)

    with torch.cuda.device(target_device):
        output_gpu = torch.empty(
            (total_seq_len, 8, 128),
            dtype=torch.bfloat16,
            device=target_device,
        )
        new_state_gpu = torch.empty(
            (num_seqs, 8, 128, 128),
            dtype=torch.float32,
            device=target_device,
        )

        if num_seqs > 0:
            average_seq_len = max(
                1,
                total_seq_len // num_seqs,
            )
            seq_len_bucket = _next_power_of_two(
                average_seq_len
            )
            num_seqs_bucket = _next_power_of_two(num_seqs)

            grid = lambda meta: (
                num_seqs,
                4,
                128 // meta["BLOCK_V"],
            )

            _gdn_prefill_paired_kernel[grid](
                q_gpu,
                k_gpu,
                v_gpu,
                state_gpu if state_gpu is not None else q_gpu,
                A_log_gpu,
                a_gpu,
                dt_bias_gpu,
                b_gpu,
                cu_seqlens_gpu,
                output_gpu,
                new_state_gpu,
                scale_value,
                seq_len_bucket,
                num_seqs_bucket,
                HEAD_SIZE=128,
                NUM_Q_HEADS=4,
                NUM_K_HEADS=4,
                NUM_V_HEADS=8,
                HAS_STATE=state_gpu is not None,
            )

    output = output_gpu.to(device=output_device)
    new_state = new_state_gpu.to(device=state_output_device)
    return output, new_state