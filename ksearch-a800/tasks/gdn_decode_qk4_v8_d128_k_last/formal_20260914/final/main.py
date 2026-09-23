import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_decode_independent_rows_kernel(
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
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    row_blocks = 128 // BLOCK_V

    row_block = pid % row_blocks
    head_index = pid // row_blocks
    value_head = head_index % 8
    batch_index = head_index // 8
    query_head = value_head // 2

    row_offsets = row_block * BLOCK_V + tl.arange(0, BLOCK_V)
    row_offsets = tl.max_contiguous(
        tl.multiple_of(row_offsets, BLOCK_V),
        BLOCK_V,
    )

    key_offsets = tl.arange(0, 128)
    key_offsets = tl.max_contiguous(
        tl.multiple_of(key_offsets, 128),
        128,
    )

    q_base = (batch_index * 4 + query_head) * 128
    value_base = (batch_index * 8 + value_head) * 128
    state_base = (batch_index * 8 + value_head) * 16384

    q_values = tl.load(
        q_ptr + q_base + key_offsets
    ).to(tl.float32)
    k_values = tl.load(
        k_ptr + q_base + key_offsets
    ).to(tl.float32)
    v_values = tl.load(
        v_ptr + value_base + row_offsets
    ).to(tl.float32)

    state_row_offsets = state_base + row_offsets * 128
    state_row_offsets = tl.multiple_of(state_row_offsets, 128)
    state_offsets = (
        state_row_offsets[:, None]
        + key_offsets[None, :]
    )

    state_values = tl.load(
        state_ptr + state_offsets,
        cache_modifier=".cg",
    )

    r_values = tl.sum(
        state_values * k_values[None, :],
        axis=1,
    )
    z_values = tl.sum(
        state_values * q_values[None, :],
        axis=1,
    )
    kq_value = tl.sum(
        k_values * q_values,
        axis=0,
    )

    gate_index = batch_index * 8 + value_head
    decay_input = (
        tl.load(a_ptr + gate_index).to(tl.float32)
        + tl.load(dt_bias_ptr + value_head)
    )
    abs_decay_input = tl.abs(decay_input)
    softplus_value = (
        tl.maximum(decay_input, 0.0)
        + tl.log(1.0 + tl.exp(-abs_decay_input))
    )

    A_log_value = tl.load(A_log_ptr + value_head)
    decay = tl.exp(
        -tl.exp(A_log_value) * softplus_value
    )

    beta_input = tl.load(
        b_ptr + gate_index
    ).to(tl.float32)
    beta = 1.0 / (1.0 + tl.exp(-beta_input))

    delta = beta * (
        v_values - decay * r_values
    )
    updated_state = (
        decay * state_values
        + delta[:, None] * k_values[None, :]
    )

    tl.store(
        new_state_ptr + state_offsets,
        updated_state,
        cache_modifier=".cs",
    )

    output_values = scale * (
        decay * z_values + delta * kq_value
    )
    tl.store(
        output_ptr + value_base + row_offsets,
        output_values,
    )


def _validate_tensor(
    tensor,
    name,
    expected_shape,
    expected_dtype,
):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(tensor.shape) != tuple(expected_shape):
        raise ValueError(
            f"{name} must have shape {tuple(expected_shape)}, "
            f"but got {tuple(tensor.shape)}"
        )
    if tensor.dtype != expected_dtype:
        raise TypeError(
            f"{name} must have dtype {expected_dtype}, "
            f"but got {tensor.dtype}"
        )


def _move_to_cuda(tensor, device):
    if tensor.device.type == "cuda":
        if tensor.device == device:
            return tensor if tensor.is_contiguous() else tensor.contiguous()
        return tensor.to(device=device).contiguous()
    return tensor.cuda(device=device).contiguous()


def _restore_device(tensor, device):
    if tensor.device == device:
        return tensor
    return tensor.to(device)


@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to execute "
            "gdn_decode_qk4_v8_d128_k_last"
        )

    if q.ndim != 4:
        raise ValueError("q must have rank 4")

    batch_size = q.shape[0]

    _validate_tensor(
        q,
        "q",
        (batch_size, 1, 4, 128),
        torch.bfloat16,
    )
    _validate_tensor(
        k,
        "k",
        (batch_size, 1, 4, 128),
        torch.bfloat16,
    )
    _validate_tensor(
        v,
        "v",
        (batch_size, 1, 8, 128),
        torch.bfloat16,
    )
    _validate_tensor(
        A_log,
        "A_log",
        (8,),
        torch.float32,
    )
    _validate_tensor(
        a,
        "a",
        (batch_size, 1, 8),
        torch.bfloat16,
    )
    _validate_tensor(
        dt_bias,
        "dt_bias",
        (8,),
        torch.float32,
    )
    _validate_tensor(
        b,
        "b",
        (batch_size, 1, 8),
        torch.bfloat16,
    )

    if state is not None:
        _validate_tensor(
            state,
            "state",
            (batch_size, 8, 128, 128),
            torch.float32,
        )

    default_scale = 1.0 / math.sqrt(128.0)
    if scale is None:
        scale_value = default_scale
    elif isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            raise ValueError(
                "scale tensor must contain exactly one element"
            )
        scale_value = float(scale.detach().item())
        if scale_value == 0.0:
            scale_value = default_scale
    else:
        scale_value = float(scale)
        if scale_value == 0.0:
            scale_value = default_scale

    if q.device.type == "cuda":
        target_device = q.device
    else:
        target_device = None
        for tensor in (
            k,
            v,
            state,
            A_log,
            a,
            dt_bias,
            b,
        ):
            if (
                isinstance(tensor, torch.Tensor)
                and tensor.device.type == "cuda"
            ):
                target_device = tensor.device
                break

        if target_device is None:
            target_device = torch.device(
                "cuda",
                torch.cuda.current_device(),
            )

    q_original_device = q.device
    state_original_device = (
        state.device if state is not None else q_original_device
    )

    with torch.cuda.device(target_device):
        q_cuda = _move_to_cuda(q, target_device)
        k_cuda = _move_to_cuda(k, target_device)
        v_cuda = _move_to_cuda(v, target_device)
        A_log_cuda = _move_to_cuda(A_log, target_device)
        a_cuda = _move_to_cuda(a, target_device)
        dt_bias_cuda = _move_to_cuda(dt_bias, target_device)
        b_cuda = _move_to_cuda(b, target_device)

        if state is None:
            state_cuda = torch.zeros(
                (batch_size, 8, 128, 128),
                dtype=torch.float32,
                device=target_device,
            )
        else:
            state_cuda = _move_to_cuda(state, target_device)

        output_cuda = torch.empty(
            (batch_size, 1, 8, 128),
            dtype=torch.bfloat16,
            device=target_device,
        )
        new_state_cuda = torch.empty(
            (batch_size, 8, 128, 128),
            dtype=torch.float32,
            device=target_device,
        )

        if batch_size > 0:
            block_v = 32
            grid = (
                batch_size * 8 * (128 // block_v),
            )

            _gdn_decode_independent_rows_kernel[grid](
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
                BLOCK_V=block_v,
                num_warps=4,
                num_stages=2,
            )

    output = _restore_device(
        output_cuda,
        q_original_device,
    )
    new_state = _restore_device(
        new_state_cuda,
        state_original_device,
    )
    return output, new_state