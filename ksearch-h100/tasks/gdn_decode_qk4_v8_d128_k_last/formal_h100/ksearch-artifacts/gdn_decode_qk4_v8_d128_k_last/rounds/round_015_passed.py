# solution=GPT-5.6-Sol_gdn_decode_qk4_v8_d128_k_last_triton_optimized_r15 score=438.92664213030463 passed=True
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_grouped_row_streaming_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    state_ptr,
    A_log_ptr,
    a_ptr,
    dt_bias_ptr,
    beta_input_ptr,
    output_ptr,
    new_state_ptr,
    scale,
    HAS_STATE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)

    groups_per_head = 128 // BLOCK_ROWS
    row_group = pid % groups_per_head
    batch_head = pid // groups_per_head

    value_head = batch_head % 8
    batch = batch_head // 8
    query_head = value_head // 2

    rows = row_group * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    k_offsets = tl.arange(0, BLOCK_K)

    qk_offset = (batch * 4 + query_head) * 128 + k_offsets
    state_offset = (
        ((batch * 8 + value_head) * 128 + rows[:, None]) * 128
        + k_offsets[None, :]
    )
    gate_offset = batch * 8 + value_head
    value_offset = (batch * 8 + value_head) * 128 + rows

    q_values = tl.load(q_ptr + qk_offset).to(tl.float32)
    k_values = tl.load(k_ptr + qk_offset).to(tl.float32)

    if HAS_STATE:
        state_rows = tl.load(state_ptr + state_offset).to(tl.float32)
    else:
        state_rows = tl.zeros((BLOCK_ROWS, BLOCK_K), dtype=tl.float32)

    a_value = tl.load(a_ptr + gate_offset).to(tl.float32)
    dt_value = tl.load(dt_bias_ptr + value_head).to(tl.float32)
    A_value = tl.load(A_log_ptr + value_head).to(tl.float32)
    beta_input = tl.load(beta_input_ptr + gate_offset).to(tl.float32)
    v_values = tl.load(v_ptr + value_offset).to(tl.float32)

    x = a_value + dt_value
    softplus_x = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
    decay = tl.exp(-tl.exp(A_value) * softplus_x)
    beta = 1.0 / (1.0 + tl.exp(-beta_input))

    state_dot_k = tl.sum(state_rows * k_values[None, :], axis=1)
    state_dot_q = tl.sum(state_rows * q_values[None, :], axis=1)
    q_dot_k = tl.sum(q_values * k_values, axis=0)

    delta = beta * (v_values - decay * state_dot_k)
    updated_rows = decay * state_rows + delta[:, None] * k_values[None, :]
    output_values = scale * (decay * state_dot_q + q_dot_k * delta)

    tl.store(new_state_ptr + state_offset, updated_rows)
    tl.store(output_ptr + value_offset, output_values)


def _to_cuda_tensor(tensor, device, name):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.is_cuda:
        return tensor.to(device=device).contiguous()
    return tensor.cuda(device=device).contiguous()


@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    tensor_inputs = {
        "q": q,
        "k": k,
        "v": v,
        "A_log": A_log,
        "a": a,
        "dt_bias": dt_bias,
        "b": b,
    }
    if state is not None:
        tensor_inputs["state"] = state

    for name, tensor in tensor_inputs.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to execute the Triton GDN kernel")

    cuda_devices = [
        tensor.device for tensor in tensor_inputs.values() if tensor.is_cuda
    ]
    device = (
        q.device
        if q.is_cuda
        else cuda_devices[0]
        if cuda_devices
        else torch.device("cuda")
    )

    q_device = q.device
    state_device = state.device if state is not None else q_device

    q_cuda = _to_cuda_tensor(q, device, "q")
    k_cuda = _to_cuda_tensor(k, device, "k")
    v_cuda = _to_cuda_tensor(v, device, "v")
    A_log_cuda = _to_cuda_tensor(A_log, device, "A_log")
    a_cuda = _to_cuda_tensor(a, device, "a")
    dt_bias_cuda = _to_cuda_tensor(dt_bias, device, "dt_bias")
    b_cuda = _to_cuda_tensor(b, device, "b")

    has_state = state is not None
    state_cuda = (
        _to_cuda_tensor(state, device, "state") if has_state else q_cuda
    )

    batch_size = q.shape[0]
    output_cuda = torch.empty(
        (batch_size, 1, 8, 128),
        device=device,
        dtype=torch.bfloat16,
    )
    new_state_cuda = torch.empty(
        (batch_size, 8, 128, 128),
        device=device,
        dtype=torch.float32,
    )

    if scale is None:
        scale_value = 1.0 / math.sqrt(128.0)
    elif isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            raise ValueError("scale must be a scalar")
        scale_value = float(scale.detach().item())
        if scale_value == 0.0:
            scale_value = 1.0 / math.sqrt(128.0)
    else:
        scale_value = float(scale)
        if scale_value == 0.0:
            scale_value = 1.0 / math.sqrt(128.0)

    if batch_size == 1:
        block_rows = 4
        num_warps = 4
    elif batch_size < 32:
        block_rows = 8
        num_warps = 8
    else:
        block_rows = 16
        num_warps = 4

    grid = (batch_size * 8 * (128 // block_rows),)

    with torch.cuda.device(device):
        _gdn_grouped_row_streaming_kernel[grid](
            q_cuda,
            k_cuda,
            v_cuda,
            state_cuda,
            A_log_cuda,
            a_cuda,
            dt_bias_cuda,
            b_cuda,
            output_cuda,
            new_state_cuda,
            scale_value,
            HAS_STATE=has_state,
            BLOCK_K=128,
            BLOCK_ROWS=block_rows,
            num_warps=num_warps,
        )

    output = output_cuda.to(device=q_device)
    new_state = new_state_cuda.to(device=state_device)
    return output, new_state