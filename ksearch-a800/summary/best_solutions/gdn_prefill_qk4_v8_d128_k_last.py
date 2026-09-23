# task: gdn_prefill_qk4_v8_d128_k_last
# bench: FlashInfer | batch: formal_20260914
# final eval (official evaluator, full workloads): valid=True pass=100/100 geomean=195.329x
# feedback best (5-workload sample during search): 227.748x
# torch fallback audit: 干净 (-)
# tokens: 1,869,496

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_single_token_kernel(
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
    NUM_V_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    value_head = tl.program_id(1)
    value_block = tl.program_id(2)

    row_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    k_offsets = tl.arange(0, HEAD_SIZE)
    row_mask = row_offsets < HEAD_SIZE

    seq_start = tl.load(cu_seqlens_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_ptr + seq_idx + 1)
    active = seq_end > seq_start

    query_head = value_head // 2
    qk_base = seq_start * (4 * HEAD_SIZE) + query_head * HEAD_SIZE

    q_values = tl.load(
        q_ptr + qk_base + k_offsets,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    k_values = tl.load(
        k_ptr + qk_base + k_offsets,
        mask=active,
        other=0.0,
    ).to(tl.float32)

    gate_offset = seq_start * NUM_V_HEADS + value_head
    a_value = tl.load(
        a_ptr + gate_offset,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    b_value = tl.load(
        b_ptr + gate_offset,
        mask=active,
        other=0.0,
    ).to(tl.float32)

    dt_value = tl.load(dt_bias_ptr + value_head).to(tl.float32)
    decay_rate = tl.exp(
        tl.load(A_log_ptr + value_head).to(tl.float32)
    )

    gate_input = a_value + dt_value
    softplus_value = (
        tl.maximum(gate_input, 0.0)
        + tl.log(1.0 + tl.exp(-tl.abs(gate_input)))
    )
    decay = tl.exp(-decay_rate * softplus_value)
    beta = 1.0 / (1.0 + tl.exp(-b_value))

    state_base = (
        (seq_idx * NUM_V_HEADS + value_head)
        * HEAD_SIZE
        * HEAD_SIZE
    )
    state_offsets = (
        state_base
        + row_offsets[:, None] * HEAD_SIZE
        + k_offsets[None, :]
    )

    state_values = tl.load(
        state_ptr + state_offsets,
        mask=active & row_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    state_values *= decay
    old_value = tl.sum(
        state_values * k_values[None, :],
        axis=1,
    )

    value_offsets = (
        seq_start * NUM_V_HEADS * HEAD_SIZE
        + value_head * HEAD_SIZE
        + row_offsets
    )
    input_value = tl.load(
        v_ptr + value_offsets,
        mask=active & row_mask,
        other=0.0,
    ).to(tl.float32)

    state_values += (
        beta * (input_value - old_value)
    )[:, None] * k_values[None, :]

    tl.store(
        new_state_ptr + state_offsets,
        state_values,
        mask=row_mask[:, None],
    )

    output_values = scale * tl.sum(
        state_values * q_values[None, :],
        axis=1,
    )
    tl.store(
        output_ptr + value_offsets,
        output_values,
        mask=active & row_mask,
    )


@triton.jit
def _gdn_recurrent_kernel(
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
    NUM_V_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    value_head = tl.program_id(1)
    value_block = tl.program_id(2)

    row_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    k_offsets = tl.arange(0, HEAD_SIZE)
    row_mask = row_offsets < HEAD_SIZE

    seq_start = tl.load(cu_seqlens_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_ptr + seq_idx + 1)
    active = seq_end > seq_start

    state_base = (
        (seq_idx * NUM_V_HEADS + value_head)
        * HEAD_SIZE
        * HEAD_SIZE
    )
    state_offsets = (
        state_base
        + row_offsets[:, None] * HEAD_SIZE
        + k_offsets[None, :]
    )

    state_values = tl.load(
        state_ptr + state_offsets,
        mask=active & row_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    decay_rate = tl.exp(
        tl.load(A_log_ptr + value_head).to(tl.float32)
    )
    dt_value = tl.load(dt_bias_ptr + value_head).to(tl.float32)
    query_head = value_head // 2

    for token_idx in tl.range(seq_start, seq_end):
        qk_base = (
            token_idx * (4 * HEAD_SIZE)
            + query_head * HEAD_SIZE
        )
        q_values = tl.load(
            q_ptr + qk_base + k_offsets
        ).to(tl.float32)
        k_values = tl.load(
            k_ptr + qk_base + k_offsets
        ).to(tl.float32)

        gate_offset = token_idx * NUM_V_HEADS + value_head
        a_value = tl.load(a_ptr + gate_offset).to(tl.float32)
        b_value = tl.load(b_ptr + gate_offset).to(tl.float32)

        gate_input = a_value + dt_value
        softplus_value = (
            tl.maximum(gate_input, 0.0)
            + tl.log(1.0 + tl.exp(-tl.abs(gate_input)))
        )
        decay = tl.exp(-decay_rate * softplus_value)
        beta = 1.0 / (1.0 + tl.exp(-b_value))

        state_values *= decay
        old_value = tl.sum(
            state_values * k_values[None, :],
            axis=1,
        )

        value_offsets = (
            token_idx * NUM_V_HEADS * HEAD_SIZE
            + value_head * HEAD_SIZE
            + row_offsets
        )
        input_value = tl.load(
            v_ptr + value_offsets,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)

        state_values += (
            beta * (input_value - old_value)
        )[:, None] * k_values[None, :]

        output_values = scale * tl.sum(
            state_values * q_values[None, :],
            axis=1,
        )
        tl.store(
            output_ptr + value_offsets,
            output_values,
            mask=row_mask,
        )

    tl.store(
        new_state_ptr + state_offsets,
        state_values,
        mask=row_mask[:, None],
    )


@torch.no_grad()
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
    named_tensors = {
        "q": q,
        "k": k,
        "v": v,
        "A_log": A_log,
        "a": a,
        "dt_bias": dt_bias,
        "b": b,
        "cu_seqlens": cu_seqlens,
    }
    if state is not None:
        named_tensors["state"] = state

    for name, value in named_tensors.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to execute the Triton GDN kernel")

    if q.ndim != 3 or tuple(q.shape[1:]) != (4, 128):
        raise ValueError("q must have shape [total_seq_len, 4, 128]")
    if k.ndim != 3 or tuple(k.shape[1:]) != (4, 128):
        raise ValueError("k must have shape [total_seq_len, 4, 128]")
    if v.ndim != 3 or tuple(v.shape[1:]) != (8, 128):
        raise ValueError("v must have shape [total_seq_len, 8, 128]")
    if a.ndim != 2 or a.shape[1] != 8:
        raise ValueError("a must have shape [total_seq_len, 8]")
    if b.ndim != 2 or b.shape[1] != 8:
        raise ValueError("b must have shape [total_seq_len, 8]")
    if A_log.ndim != 1 or A_log.numel() != 8:
        raise ValueError("A_log must have shape [8]")
    if dt_bias.ndim != 1 or dt_bias.numel() != 8:
        raise ValueError("dt_bias must have shape [8]")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 1:
        raise ValueError(
            "cu_seqlens must be a nonempty one-dimensional tensor"
        )

    total_seq_len = q.shape[0]
    num_seqs = cu_seqlens.numel() - 1

    if k.shape[0] != total_seq_len:
        raise ValueError("q and k must have the same total sequence length")
    if v.shape[0] != total_seq_len:
        raise ValueError("q and v must have the same total sequence length")
    if a.shape[0] != total_seq_len or b.shape[0] != total_seq_len:
        raise ValueError("a and b must match total_seq_len")

    if q.dtype != torch.bfloat16:
        raise TypeError("q must have dtype torch.bfloat16")
    if k.dtype != torch.bfloat16:
        raise TypeError("k must have dtype torch.bfloat16")
    if v.dtype != torch.bfloat16:
        raise TypeError("v must have dtype torch.bfloat16")
    if a.dtype != torch.bfloat16:
        raise TypeError("a must have dtype torch.bfloat16")
    if b.dtype != torch.bfloat16:
        raise TypeError("b must have dtype torch.bfloat16")
    if A_log.dtype != torch.float32:
        raise TypeError("A_log must have dtype torch.float32")
    if dt_bias.dtype != torch.float32:
        raise TypeError("dt_bias must have dtype torch.float32")
    if state is not None and state.dtype != torch.float32:
        raise TypeError("state must have dtype torch.float32")
    if cu_seqlens.dtype != torch.int64:
        raise TypeError("cu_seqlens must have dtype torch.int64")

    if state is not None and tuple(state.shape) != (
        num_seqs,
        8,
        128,
        128,
    ):
        raise ValueError(
            "state must have shape [num_seqs, 8, 128, 128]"
        )

    output_device = q.device
    new_state_device = state.device if state is not None else q.device

    cuda_device = None
    for value in named_tensors.values():
        if value.is_cuda:
            cuda_device = value.device
            break

    if cuda_device is None:
        cuda_device = torch.device("cuda", torch.cuda.current_device())

    def move_tensor(value):
        if value.device != cuda_device:
            value = value.cuda(device=cuda_device)
        if not value.is_contiguous():
            value = value.contiguous()
        return value

    if scale is None:
        scale_value = 1.0 / math.sqrt(128.0)
    elif isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            raise ValueError("scale must be a scalar")
        scale_value = float(scale.detach().cpu().item())
    else:
        scale_value = float(scale)

    if scale_value == 0.0:
        scale_value = 1.0 / math.sqrt(128.0)

    with torch.cuda.device(cuda_device):
        q_gpu = move_tensor(q)
        k_gpu = move_tensor(k)
        v_gpu = move_tensor(v)
        A_log_gpu = move_tensor(A_log)
        a_gpu = move_tensor(a)
        dt_bias_gpu = move_tensor(dt_bias)
        b_gpu = move_tensor(b)
        cu_gpu = move_tensor(cu_seqlens)

        if state is None:
            state_gpu = torch.zeros(
                (num_seqs, 8, 128, 128),
                dtype=torch.float32,
                device=cuda_device,
            )
        else:
            state_gpu = move_tensor(state)

        output_gpu = torch.empty(
            (total_seq_len, 8, 128),
            dtype=torch.bfloat16,
            device=cuda_device,
        )
        new_state_gpu = torch.empty(
            (num_seqs, 8, 128, 128),
            dtype=torch.float32,
            device=cuda_device,
        )

        if num_seqs > 0:
            grid = (num_seqs, 8, 4)

            if total_seq_len <= 1:
                _gdn_single_token_kernel[grid](
                    q_gpu,
                    k_gpu,
                    v_gpu,
                    state_gpu,
                    A_log_gpu,
                    a_gpu,
                    dt_bias_gpu,
                    b_gpu,
                    cu_gpu,
                    output_gpu,
                    new_state_gpu,
                    scale_value,
                    NUM_V_HEADS=8,
                    HEAD_SIZE=128,
                    BLOCK_V=32,
                    num_warps=8,
                )
            else:
                _gdn_recurrent_kernel[grid](
                    q_gpu,
                    k_gpu,
                    v_gpu,
                    state_gpu,
                    A_log_gpu,
                    a_gpu,
                    dt_bias_gpu,
                    b_gpu,
                    cu_gpu,
                    output_gpu,
                    new_state_gpu,
                    scale_value,
                    NUM_V_HEADS=8,
                    HEAD_SIZE=128,
                    BLOCK_V=32,
                    num_warps=8,
                )

    output = output_gpu.to(device=output_device)
    new_state = new_state_gpu.to(device=new_state_device)
    return output, new_state