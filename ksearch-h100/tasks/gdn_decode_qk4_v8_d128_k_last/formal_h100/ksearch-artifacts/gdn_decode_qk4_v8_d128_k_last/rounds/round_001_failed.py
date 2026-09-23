# solution=GPT-5.6-Sol_gdn_decode_qk4_v8_d128_k_last_triton_optimized_r1 score=283.77946455333534 passed=False
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_row_streaming_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    state_ptr,
    A_log_ptr,
    a_ptr,
    dt_bias_ptr,
    b_ptr,
    output_ptr,
    new_state_ptr,
    scale,
    HEAD_SIZE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
):
    pid = tl.program_id(0)

    row = pid % HEAD_SIZE
    bh = pid // HEAD_SIZE
    v_head = bh % NUM_V_HEADS
    batch = bh // NUM_V_HEADS
    q_head = v_head // (NUM_V_HEADS // NUM_Q_HEADS)

    offsets = tl.arange(0, HEAD_SIZE)

    q_base = (batch * NUM_Q_HEADS + q_head) * HEAD_SIZE
    k_base = q_base
    state_base = (
        ((batch * NUM_V_HEADS + v_head) * HEAD_SIZE + row) * HEAD_SIZE
    )

    q_values = tl.load(q_ptr + q_base + offsets).to(tl.float32)
    k_values = tl.load(k_ptr + k_base + offsets).to(tl.float32)
    state_values = tl.load(state_ptr + state_base + offsets).to(tl.float32)

    state_dot_k = tl.sum(state_values * k_values, axis=0)
    state_dot_q = tl.sum(state_values * q_values, axis=0)
    q_dot_k = tl.sum(q_values * k_values, axis=0)

    head_offset = batch * NUM_V_HEADS + v_head
    gate_input = (
        tl.load(a_ptr + head_offset).to(tl.float32)
        + tl.load(dt_bias_ptr + v_head).to(tl.float32)
    )
    softplus = tl.maximum(gate_input, 0.0) + tl.log(
        1.0 + tl.exp(-tl.abs(gate_input))
    )
    decay = tl.exp(
        -tl.exp(tl.load(A_log_ptr + v_head).to(tl.float32)) * softplus
    )

    beta_input = tl.load(b_ptr + head_offset).to(tl.float32)
    beta = 1.0 / (1.0 + tl.exp(-beta_input))
    value = tl.load(v_ptr + head_offset * HEAD_SIZE + row).to(tl.float32)

    delta = beta * (value - decay * state_dot_k)
    updated_state = decay * state_values + delta * k_values
    result = scale * (decay * state_dot_q + delta * q_dot_k)

    tl.store(new_state_ptr + state_base + offsets, updated_state)
    tl.store(output_ptr + head_offset * HEAD_SIZE + row, result)


def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    if not torch.cuda.is_available():
        raise RuntimeError("gdn_decode_qk4_v8_d128_k_last requires CUDA")

    tensors = {
        "q": q,
        "k": k,
        "v": v,
        "A_log": A_log,
        "a": a,
        "dt_bias": dt_bias,
        "b": b,
    }
    if state is not None:
        tensors["state"] = state

    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    cuda_devices = {
        tensor.device
        for tensor in tensors.values()
        if tensor.device.type == "cuda"
    }
    if len(cuda_devices) > 1:
        devices = ", ".join(sorted(str(device) for device in cuda_devices))
        raise ValueError(f"all GPU inputs must be on one CUDA device, got {devices}")

    target_device = (
        next(iter(cuda_devices))
        if cuda_devices
        else torch.device("cuda", torch.cuda.current_device())
    )

    original_output_device = q.device
    original_state_device = state.device if state is not None else q.device

    def to_cuda(tensor):
        if tensor.device == target_device:
            return tensor
        if tensor.device.type == "cuda":
            return tensor.to(target_device)
        return tensor.cuda(device=target_device)

    q_gpu = to_cuda(q).contiguous()
    k_gpu = to_cuda(k).contiguous()
    v_gpu = to_cuda(v).contiguous()
    A_log_gpu = to_cuda(A_log).contiguous()
    a_gpu = to_cuda(a).contiguous()
    dt_bias_gpu = to_cuda(dt_bias).contiguous()
    b_gpu = to_cuda(b).contiguous()

    if q_gpu.ndim != 4 or tuple(q_gpu.shape[1:]) != (1, 4, 128):
        raise ValueError("q must have shape [batch_size, 1, 4, 128]")
    batch_size = q_gpu.shape[0]
    if tuple(k_gpu.shape) != (batch_size, 1, 4, 128):
        raise ValueError("k must have shape [batch_size, 1, 4, 128]")
    if tuple(v_gpu.shape) != (batch_size, 1, 8, 128):
        raise ValueError("v must have shape [batch_size, 1, 8, 128]")
    if tuple(a_gpu.shape) != (batch_size, 1, 8):
        raise ValueError("a must have shape [batch_size, 1, 8]")
    if tuple(b_gpu.shape) != (batch_size, 1, 8):
        raise ValueError("b must have shape [batch_size, 1, 8]")
    if tuple(A_log_gpu.shape) != (8,):
        raise ValueError("A_log must have shape [8]")
    if tuple(dt_bias_gpu.shape) != (8,):
        raise ValueError("dt_bias must have shape [8]")

    if state is None:
        state_gpu = torch.zeros(
            (batch_size, 8, 128, 128),
            device=target_device,
            dtype=torch.float32,
        )
    else:
        state_gpu = to_cuda(state).contiguous()
        if tuple(state_gpu.shape) != (batch_size, 8, 128, 128):
            raise ValueError(
                "state must have shape [batch_size, 8, 128, 128]"
            )

    if scale is None:
        scale_value = 1.0 / math.sqrt(128)
    elif isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            raise ValueError("scale must be a scalar")
        scale_value = float(scale.detach().item())
    else:
        scale_value = float(scale)

    if scale_value == 0.0:
        scale_value = 1.0 / math.sqrt(128)

    output_gpu = torch.empty(
        (batch_size, 1, 8, 128),
        device=target_device,
        dtype=torch.bfloat16,
    )
    new_state_gpu = torch.empty(
        (batch_size, 8, 128, 128),
        device=target_device,
        dtype=torch.float32,
    )

    grid = (batch_size * 8 * 128,)
    with torch.cuda.device(target_device):
        _gdn_row_streaming_kernel[grid](
            q_gpu,
            k_gpu,
            v_gpu,
            state_gpu,
            A_log_gpu,
            a_gpu,
            dt_bias_gpu,
            b_gpu,
            output_gpu,
            new_state_gpu,
            scale_value,
            HEAD_SIZE=128,
            NUM_Q_HEADS=4,
            NUM_V_HEADS=8,
            num_warps=4,
        )

    output = output_gpu.to(original_output_device)
    new_state = new_state_gpu.to(original_state_device)
    return output, new_state