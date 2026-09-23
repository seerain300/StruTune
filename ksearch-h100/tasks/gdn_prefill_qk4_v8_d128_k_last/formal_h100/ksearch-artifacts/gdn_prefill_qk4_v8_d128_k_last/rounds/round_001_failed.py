# solution=GPT-5.6-Sol_gdn_prefill_qk4_v8_d128_k_last_triton_optimized_r1 score=233.47315736084516 passed=False
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_prefill_direct_kernel(
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
    HEAD_SIZE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    HAS_STATE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    value_head = tl.program_id(1)
    value_block = tl.program_id(2)

    value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    key_offsets = tl.arange(0, HEAD_SIZE)
    value_mask = value_offsets < HEAD_SIZE

    query_head = value_head // (NUM_V_HEADS // NUM_Q_HEADS)
    key_head = value_head // (NUM_V_HEADS // NUM_K_HEADS)

    seq_start = tl.load(cu_seqlens_ptr + seq_idx).to(tl.int64)
    seq_end = tl.load(cu_seqlens_ptr + seq_idx + 1).to(tl.int64)

    state_offsets = (
        ((seq_idx * NUM_V_HEADS + value_head) * HEAD_SIZE + value_offsets[:, None])
        * HEAD_SIZE
        + key_offsets[None, :]
    )
    state_mask = value_mask[:, None]

    if HAS_STATE:
        state_values = tl.load(
            state_ptr + state_offsets,
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)
    else:
        state_values = tl.zeros((BLOCK_V, HEAD_SIZE), dtype=tl.float32)

    decay_rate = tl.exp(tl.load(A_log_ptr + value_head).to(tl.float32))
    decay_bias = tl.load(dt_bias_ptr + value_head).to(tl.float32)

    token_idx = seq_start
    while token_idx < seq_end:
        key_base = (token_idx * NUM_K_HEADS + key_head) * HEAD_SIZE
        query_base = (token_idx * NUM_Q_HEADS + query_head) * HEAD_SIZE
        value_base = (token_idx * NUM_V_HEADS + value_head) * HEAD_SIZE

        key_values = tl.load(k_ptr + key_base + key_offsets).to(tl.float32)
        query_values = tl.load(q_ptr + query_base + key_offsets).to(tl.float32)
        input_values = tl.load(
            v_ptr + value_base + value_offsets,
            mask=value_mask,
            other=0.0,
        ).to(tl.float32)

        gate_offset = token_idx * NUM_V_HEADS + value_head
        decay_input = (
            tl.load(a_ptr + gate_offset).to(tl.float32) + decay_bias
        )
        softplus = tl.maximum(decay_input, 0.0) + tl.log(
            1.0 + tl.exp(-tl.abs(decay_input))
        )
        decay = tl.exp(-decay_rate * softplus)
        beta = tl.sigmoid(tl.load(b_ptr + gate_offset).to(tl.float32))

        decayed_state = state_values * decay
        predicted_values = tl.sum(
            decayed_state * key_values[None, :],
            axis=1,
        )
        correction = beta * (input_values - predicted_values)
        state_values = (
            decayed_state
            + correction[:, None] * key_values[None, :]
        )

        output_values = (
            tl.sum(state_values * query_values[None, :], axis=1) * scale
        )
        output_offsets = (
            (token_idx * NUM_V_HEADS + value_head) * HEAD_SIZE
            + value_offsets
        )
        tl.store(
            output_ptr + output_offsets,
            output_values,
            mask=value_mask,
        )

        token_idx += 1

    tl.store(
        new_state_ptr + state_offsets,
        state_values,
        mask=state_mask,
    )


def _validate_tensor(name, tensor, dtype, ndim):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {tensor.ndim}")


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
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

    if q.shape != (total_seq_len, 4, 128):
        raise ValueError(f"q must have shape [{total_seq_len}, 4, 128]")
    if k.shape != (total_seq_len, 4, 128):
        raise ValueError(f"k must have shape [{total_seq_len}, 4, 128]")
    if v.shape != (total_seq_len, 8, 128):
        raise ValueError(f"v must have shape [{total_seq_len}, 8, 128]")
    if a.shape != (total_seq_len, 8):
        raise ValueError(f"a must have shape [{total_seq_len}, 8]")
    if b.shape != (total_seq_len, 8):
        raise ValueError(f"b must have shape [{total_seq_len}, 8]")
    if A_log.shape != (8,):
        raise ValueError("A_log must have shape [8]")
    if dt_bias.shape != (8,):
        raise ValueError("dt_bias must have shape [8]")
    if state is not None and state.shape != (num_seqs, 8, 128, 128):
        raise ValueError(
            f"state must have shape [{num_seqs}, 8, 128, 128]"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "gdn_prefill_qk4_v8_d128_k_last requires a CUDA-capable GPU"
        )

    tensors = (q, k, v, state, A_log, a, dt_bias, b, cu_seqlens)
    cuda_devices = {
        tensor.device
        for tensor in tensors
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda
    }
    if len(cuda_devices) > 1:
        raise ValueError("all CUDA input tensors must be on the same device")

    target_device = (
        next(iter(cuda_devices))
        if cuda_devices
        else torch.device("cuda", torch.cuda.current_device())
    )

    output_device = q.device
    state_output_device = state.device if state is not None else q.device

    def to_cuda(tensor):
        if tensor is None:
            return None
        if tensor.is_cuda:
            return tensor.to(device=target_device).contiguous()
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
            block_v = 4
            grid = (num_seqs, 8, triton.cdiv(128, block_v))
            _gdn_prefill_direct_kernel[grid](
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
                HEAD_SIZE=128,
                NUM_Q_HEADS=4,
                NUM_K_HEADS=4,
                NUM_V_HEADS=8,
                BLOCK_V=block_v,
                HAS_STATE=state_gpu is not None,
                num_warps=4,
                num_stages=1,
            )

    output = output_gpu.to(device=output_device)
    new_state = new_state_gpu.to(device=state_output_device)
    return output, new_state