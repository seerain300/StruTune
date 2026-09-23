# solution=GPT-5.6-Sol_gdn_prefill_qk4_v8_d128_k_last_triton_optimized_r1 score=317.8857954647292 passed=True
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_precompute_gates_kernel(
    q_ptr,
    k_ptr,
    A_log_ptr,
    a_ptr,
    dt_bias_ptr,
    b_ptr,
    gates_ptr,
    qk_ptr,
    total_seq_len,
    BLOCK_T: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    v_heads = tl.arange(0, 8)

    gate_offsets = tokens[:, None] * 8 + v_heads[None, :]
    gate_mask = tokens[:, None] < total_seq_len

    decay_input = (
        tl.load(a_ptr + gate_offsets, mask=gate_mask, other=0).to(tl.float32)
        + tl.load(dt_bias_ptr + v_heads)[None, :]
    )
    decay_rate = tl.exp(tl.load(A_log_ptr + v_heads)).to(tl.float32)[None, :]
    softplus = tl.maximum(decay_input, 0.0) + tl.log(
        1.0 + tl.exp(-tl.abs(decay_input))
    )
    decay = tl.exp(-decay_rate * softplus)

    beta_input = tl.load(
        b_ptr + gate_offsets, mask=gate_mask, other=0
    ).to(tl.float32)
    beta = tl.sigmoid(beta_input)

    tl.store(gates_ptr + gate_offsets * 2, decay, mask=gate_mask)
    tl.store(gates_ptr + gate_offsets * 2 + 1, beta, mask=gate_mask)

    qk_rows = tl.arange(0, BLOCK_T * 4)
    qk_tokens = tl.program_id(0) * BLOCK_T + qk_rows // 4
    qk_heads = qk_rows % 4
    dims = tl.arange(0, 128)

    qk_offsets = (
        (qk_tokens[:, None] * 4 + qk_heads[:, None]) * 128
        + dims[None, :]
    )
    qk_mask = qk_tokens[:, None] < total_seq_len

    queries = tl.load(
        q_ptr + qk_offsets, mask=qk_mask, other=0
    ).to(tl.float32)
    keys = tl.load(
        k_ptr + qk_offsets, mask=qk_mask, other=0
    ).to(tl.float32)

    qk = tl.sum(queries * keys, axis=1)
    tl.store(
        qk_ptr + qk_tokens * 4 + qk_heads,
        qk,
        mask=qk_tokens < total_seq_len,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_V": 1}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_V": 1}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_V": 2}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_V": 2}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_V": 4}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_V": 4}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_V": 4}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_V": 8}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_V": 8}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_V": 8}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_V": 16}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_V": 16}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_V": 16}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_V": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_V": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_V": 32}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_V": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_V": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_V": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_V": 128}, num_warps=8, num_stages=1),
    ],
    key=[
        "SEQ_LEN_BUCKET",
        "NUM_SEQS_BUCKET",
        "HAS_STATE",
        "PRECOMPUTED_GATES",
    ],
)
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
    gates_ptr,
    qk_ptr,
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
    PRECOMPUTED_GATES: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    value_head = tl.program_id(1)
    value_block = tl.program_id(2)

    value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    key_offsets = tl.arange(0, HEAD_SIZE)

    query_head = value_head // (NUM_V_HEADS // NUM_Q_HEADS)
    key_head = value_head // (NUM_V_HEADS // NUM_K_HEADS)

    seq_start = tl.load(cu_seqlens_ptr + seq_idx).to(tl.int32)
    seq_end = tl.load(cu_seqlens_ptr + seq_idx + 1).to(tl.int32)

    state_offsets = (
        (
            (seq_idx * NUM_V_HEADS + value_head) * HEAD_SIZE
            + value_offsets[:, None]
        )
        * HEAD_SIZE
        + key_offsets[None, :]
    )

    if HAS_STATE:
        state_values = tl.load(
            state_ptr + state_offsets,
            cache_modifier=".cg",
        ).to(tl.float32)
    else:
        state_values = tl.zeros((BLOCK_V, HEAD_SIZE), dtype=tl.float32)

    if not PRECOMPUTED_GATES:
        decay_rate = tl.exp(
            tl.load(A_log_ptr + value_head).to(tl.float32)
        )
        decay_bias = tl.load(dt_bias_ptr + value_head).to(tl.float32)

    token_idx = seq_start
    while token_idx < seq_end:
        key_base = (token_idx * NUM_K_HEADS + key_head) * HEAD_SIZE
        query_base = (token_idx * NUM_Q_HEADS + query_head) * HEAD_SIZE
        value_base = (token_idx * NUM_V_HEADS + value_head) * HEAD_SIZE

        key_values = tl.load(
            k_ptr + key_base + key_offsets,
            cache_modifier=".ca",
        ).to(tl.float32)
        query_values = tl.load(
            q_ptr + query_base + key_offsets,
            cache_modifier=".ca",
        ).to(tl.float32)
        input_values = tl.load(
            v_ptr + value_base + value_offsets
        ).to(tl.float32)

        gate_offset = token_idx * NUM_V_HEADS + value_head

        if PRECOMPUTED_GATES:
            decay = tl.load(
                gates_ptr + gate_offset * 2,
                cache_modifier=".ca",
            ).to(tl.float32)
            beta = tl.load(
                gates_ptr + gate_offset * 2 + 1,
                cache_modifier=".ca",
            ).to(tl.float32)
            query_key = tl.load(
                qk_ptr + token_idx * NUM_Q_HEADS + query_head,
                cache_modifier=".ca",
            ).to(tl.float32)
        else:
            decay_input = (
                tl.load(a_ptr + gate_offset).to(tl.float32)
                + decay_bias
            )
            softplus = (
                tl.maximum(decay_input, 0.0)
                + tl.log(1.0 + tl.exp(-tl.abs(decay_input)))
            )
            decay = tl.exp(-decay_rate * softplus)
            beta = tl.sigmoid(
                tl.load(b_ptr + gate_offset).to(tl.float32)
            )
            query_key = tl.sum(query_values * key_values, axis=0)

        predicted_values = (
            tl.sum(state_values * key_values[None, :], axis=1) * decay
        )
        previous_output = (
            tl.sum(state_values * query_values[None, :], axis=1) * decay
        )

        correction = beta * (input_values - predicted_values)
        state_values = (
            state_values * decay
            + correction[:, None] * key_values[None, :]
        )

        output_values = (
            previous_output + correction * query_key
        ) * scale

        output_offsets = (
            (token_idx * NUM_V_HEADS + value_head) * HEAD_SIZE
            + value_offsets
        )
        tl.store(
            output_ptr + output_offsets,
            output_values,
            cache_modifier=".cs",
        )

        token_idx += 1

    tl.store(
        new_state_ptr + state_offsets,
        state_values,
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

    if num_seqs < 0:
        raise ValueError("cu_seqlens must contain at least one element")

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
    if a.shape != (total_seq_len, 8) or b.shape != (total_seq_len, 8):
        raise ValueError(
            "a and b must have shape [total_seq_len, 8]"
        )
    if A_log.shape != (8,) or dt_bias.shape != (8,):
        raise ValueError("A_log and dt_bias must have shape [8]")
    if state is not None and state.shape != (num_seqs, 8, 128, 128):
        raise ValueError(
            f"state must have shape [{num_seqs}, 8, 128, 128]"
        )

    if not torch.cuda.is_available():
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

    target_device = (
        next(iter(cuda_devices))
        if cuda_devices
        else torch.device("cuda", torch.cuda.current_device())
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
                1, total_seq_len // num_seqs
            )
            seq_len_bucket = _next_power_of_two(average_seq_len)
            num_seqs_bucket = _next_power_of_two(num_seqs)
            precomputed_gates = average_seq_len >= 64

            if precomputed_gates:
                gates_gpu = torch.empty(
                    (total_seq_len, 8, 2),
                    dtype=torch.float32,
                    device=target_device,
                )
                qk_gpu = torch.empty(
                    (total_seq_len, 4),
                    dtype=torch.float32,
                    device=target_device,
                )

                block_t = 32 if total_seq_len >= 256 else 16
                precompute_warps = 8 if block_t == 32 else 4

                _gdn_precompute_gates_kernel[
                    (triton.cdiv(total_seq_len, block_t),)
                ](
                    q_gpu,
                    k_gpu,
                    A_log_gpu,
                    a_gpu,
                    dt_bias_gpu,
                    b_gpu,
                    gates_gpu,
                    qk_gpu,
                    total_seq_len,
                    BLOCK_T=block_t,
                    num_warps=precompute_warps,
                    num_stages=1,
                )
            else:
                gates_gpu = a_gpu
                qk_gpu = q_gpu

            grid = lambda meta: (
                num_seqs,
                8,
                128 // meta["BLOCK_V"],
            )

            _gdn_prefill_direct_kernel[grid](
                q_gpu,
                k_gpu,
                v_gpu,
                state_gpu if state_gpu is not None else q_gpu,
                A_log_gpu,
                a_gpu,
                dt_bias_gpu,
                b_gpu,
                gates_gpu,
                qk_gpu,
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
                PRECOMPUTED_GATES=precomputed_gates,
            )

    return (
        output_gpu.to(device=output_device),
        new_state_gpu.to(device=state_output_device),
    )