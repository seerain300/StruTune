# solution=GPT-5.6-Sol_gdn_prefill_qk4_v8_d128_k_last_triton_optimized_r6 score=208.65849851229228 passed=True
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _prefill_kernel(
    Q, K, V, State, ALog, A, DtBias, B, CuSeqlens,
    Output, NewState,
    Scale: tl.constexpr,
    HasState: tl.constexpr,
    BlockV: tl.constexpr,
):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    v_block = tl.program_id(2)

    vr = v_block * BlockV + tl.arange(0, BlockV)
    kr = tl.arange(0, 128)

    start = tl.load(CuSeqlens + seq).to(tl.int32)
    end = tl.load(CuSeqlens + seq + 1).to(tl.int32)

    state_offsets = ((seq * 8 + head) * 128 + vr[:, None]) * 128 + kr[None, :]

    if HasState:
        state_tile = tl.load(State + state_offsets)
    else:
        state_tile = tl.full((BlockV, 128), 0.0, tl.float32)

    decay_rate = tl.exp(tl.load(ALog + head).to(tl.float32))
    bias = tl.load(DtBias + head).to(tl.float32)

    qk_base = (start * 4 + head // 2) * 128
    q_ptr = Q + qk_base
    k_ptr = K + qk_base
    v_ptr = V + (start * 8 + head) * 128
    a_ptr = A + start * 8 + head
    b_ptr = B + start * 8 + head
    out_ptr = Output + (start * 8 + head) * 128 + vr

    for _ in range(start, end):
        q_vec = tl.load(q_ptr + kr).to(tl.float32)
        k_vec = tl.load(k_ptr + kr).to(tl.float32)
        v_vec = tl.load(v_ptr + vr).to(tl.float32)

        x = tl.load(a_ptr).to(tl.float32) + bias
        softplus = tl.maximum(x, 0.0) + tl.log(
            1.0 + tl.exp(-tl.abs(x))
        )
        decay = tl.exp(-decay_rate * softplus)

        gate_input = tl.load(b_ptr).to(tl.float32)
        beta = 1.0 / (1.0 + tl.exp(-gate_input))

        decayed = decay * state_tile
        old_v = tl.sum(decayed * k_vec[None, :], axis=1)

        delta_v = beta * (v_vec - old_v)
        state_tile = decayed + k_vec[None, :] * delta_v[:, None]

        result = Scale * tl.sum(state_tile * q_vec[None, :], axis=1)
        tl.store(out_ptr, result)

        q_ptr += 512
        k_ptr += 512
        v_ptr += 1024
        a_ptr += 8
        b_ptr += 8
        out_ptr += 1024

    tl.store(NewState + state_offsets, state_tile)


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    inputs = (
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
    )

    if not torch.cuda.is_available():
        has_gpu_tensor = any(
            isinstance(x, torch.Tensor) and x.is_cuda
            for x in inputs
        )
        if has_gpu_tensor:
            raise RuntimeError(
                "CUDA tensors were provided, but CUDA is not available."
            )
        raise RuntimeError(
            "CUDA is required to run the Triton prefill kernel."
        )

    device = next(
        (
            x.device
            for x in inputs
            if isinstance(x, torch.Tensor) and x.is_cuda
        ),
        torch.device("cuda", torch.cuda.current_device()),
    )

    def on_gpu(x):
        if x is None or not isinstance(x, torch.Tensor):
            return x
        if x.device == device:
            return x
        if x.device.type == "cpu":
            return x.cuda(device=device)
        return x.to(device=device)

    (
        q_gpu,
        k_gpu,
        v_gpu,
        state_gpu,
        alog_gpu,
        a_gpu,
        bias_gpu,
        b_gpu,
        cu_gpu,
        scale_gpu,
    ) = (on_gpu(x) for x in inputs)

    num_seqs = cu_gpu.numel() - 1
    total_seq_len = q_gpu.shape[0]

    output = torch.empty(
        (total_seq_len, 8, 128),
        device=device,
        dtype=torch.bfloat16,
    )
    new_state = torch.empty(
        (num_seqs, 8, 128, 128),
        device=device,
        dtype=torch.float32,
    )

    if scale_gpu is None:
        effective_scale = 1.0 / math.sqrt(128.0)
    elif isinstance(scale_gpu, torch.Tensor):
        scale_value = float(scale_gpu.detach().item())
        effective_scale = (
            1.0 / math.sqrt(128.0)
            if scale_value == 0.0
            else scale_value
        )
    else:
        scale_value = float(scale_gpu)
        effective_scale = (
            1.0 / math.sqrt(128.0)
            if scale_value == 0.0
            else scale_value
        )

    average_seq_len = total_seq_len / max(num_seqs, 1)

    if average_seq_len <= 128.0:
        if num_seqs == 1:
            block_v = 8
        elif num_seqs <= 3:
            block_v = 16
        else:
            block_v = 32
        num_warps = 4
    else:
        block_v = 32
        if average_seq_len >= 1024.0 and num_seqs <= 4:
            num_warps = 8
        else:
            num_warps = 4

    _prefill_kernel[(num_seqs, 8, 128 // block_v)](
        q_gpu,
        k_gpu,
        v_gpu,
        state_gpu if state_gpu is not None else new_state,
        alog_gpu,
        a_gpu,
        bias_gpu,
        b_gpu,
        cu_gpu,
        output,
        new_state,
        effective_scale,
        state_gpu is not None,
        block_v,
        num_warps=num_warps,
        num_stages=1,
        enable_fp_fusion=True,
    )

    output_device = q.device
    state_device = state.device if state is not None else q.device

    return output.to(output_device), new_state.to(state_device)