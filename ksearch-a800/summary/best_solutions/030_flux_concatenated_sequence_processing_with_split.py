# task: 030_flux_concatenated_sequence_processing_with_split
# bench: SOL-L2 | batch: formal2_solL2_20260916
# final eval (official evaluator, full workloads): valid=True pass=16/16 geomean=1.379x
# feedback best (5-workload sample during search): 1.423x
# torch fallback audit: 干净 (-)
# tokens: 1,239,761

import torch
import triton
import triton.language as tl


_HIDDEN_DIM = 3072
_LOW_ROW_N64_CUTOFF = 128


@triton.jit
def _fused_logical_projection_kernel(
    hidden_ptr,
    encoder_ptr,
    weight_ptr,
    out_encoder_ptr,
    out_hidden_ptr,
    text_rows,
    hidden_rows,
    text_seq_len,
    img_seq_len,
    stride_hb,
    stride_hs,
    stride_hk,
    stride_eb,
    stride_es,
    stride_ek,
    stride_wn,
    stride_wk,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    INPUTS_CONTIGUOUS: tl.constexpr,
    ASYMMETRIC_EVICTION: tl.constexpr,
):
    pid = tl.program_id(0)

    num_text_pid_m = tl.cdiv(text_rows, BLOCK_M)
    num_hidden_pid_m = tl.cdiv(hidden_rows, BLOCK_M)
    num_pid_m = num_text_pid_m + num_hidden_pid_m
    num_pid_n = HIDDEN_DIM // BLOCK_N
    num_pid_in_group = GROUP_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)

    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + pid_in_group % group_size_m
    pid_n = pid_in_group // group_size_m

    is_text = pid_m < num_text_pid_m
    stream_pid_m = tl.where(is_text, pid_m, pid_m - num_text_pid_m)
    stream_rows = tl.where(is_text, text_rows, hidden_rows)

    stream_rows_idx = stream_pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_rows = stream_rows_idx < stream_rows

    input_base = tl.where(is_text, encoder_ptr, hidden_ptr)
    output_base = tl.where(is_text, out_encoder_ptr, out_hidden_ptr)

    if not INPUTS_CONTIGUOUS:
        sequence_len = tl.where(is_text, text_seq_len, img_seq_len)
        batch_idx = stream_rows_idx // sequence_len
        sequence_idx = stream_rows_idx - batch_idx * sequence_len

        input_stride_b = tl.where(is_text, stride_eb, stride_hb)
        input_stride_s = tl.where(is_text, stride_es, stride_hs)
        input_stride_k = tl.where(is_text, stride_ek, stride_hk)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, HIDDEN_DIM, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)

        if INPUTS_CONTIGUOUS:
            input_offsets = (
                stream_rows_idx[:, None] * HIDDEN_DIM
                + k[None, :]
            )
        else:
            input_offsets = (
                batch_idx[:, None] * input_stride_b
                + sequence_idx[:, None] * input_stride_s
                + k[None, :] * input_stride_k
            )

        if ASYMMETRIC_EVICTION:
            values = tl.load(
                input_base + input_offsets,
                mask=valid_rows[:, None],
                other=0.0,
                eviction_policy="evict_first",
            )
        else:
            values = tl.load(
                input_base + input_offsets,
                mask=valid_rows[:, None],
                other=0.0,
            )

        weight_offsets = (
            k[:, None] * stride_wk
            + cols[None, :] * stride_wn
        )

        if ASYMMETRIC_EVICTION:
            weights = tl.load(
                weight_ptr + weight_offsets,
                eviction_policy="evict_last",
            )
        else:
            weights = tl.load(weight_ptr + weight_offsets)

        accumulator = tl.dot(
            values,
            weights,
            accumulator,
            input_precision="tf32x3",
        )

    output_offsets = (
        stream_rows_idx[:, None] * HIDDEN_DIM
        + cols[None, :]
    )
    tl.store(
        output_base + output_offsets,
        accumulator,
        mask=valid_rows[:, None],
    )


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = hidden_states.shape[0]
    img_seq_len = hidden_states.shape[1]
    text_seq_len = encoder_hidden_states.shape[1]

    text_rows = batch_size * text_seq_len
    hidden_rows = batch_size * img_seq_len
    total_rows = text_rows + hidden_rows

    processed_encoder = torch.empty(
        (batch_size, text_seq_len, _HIDDEN_DIM),
        device=encoder_hidden_states.device,
        dtype=encoder_hidden_states.dtype,
    )
    processed_hidden = torch.empty(
        (batch_size, img_seq_len, _HIDDEN_DIM),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    if total_rows == 0:
        return processed_encoder, processed_hidden

    if total_rows <= 16:
        block_m, block_n, block_k = 16, 64, 32
        num_warps, num_stages = 4, 5
    elif total_rows <= 32:
        block_m, block_n, block_k = 32, 64, 32
        num_warps, num_stages = 4, 4
    elif total_rows <= _LOW_ROW_N64_CUTOFF:
        block_m, block_n, block_k = 64, 64, 32
        num_warps, num_stages = 4, 3
    else:
        block_m, block_n, block_k = 64, 128, 32
        num_warps, num_stages = 8, 3

    num_pid_m = (
        triton.cdiv(text_rows, block_m)
        + triton.cdiv(hidden_rows, block_m)
    )
    grid = (num_pid_m * triton.cdiv(_HIDDEN_DIM, block_n),)

    inputs_contiguous = (
        hidden_states.is_contiguous()
        and encoder_hidden_states.is_contiguous()
    )
    asymmetric_eviction = 65 <= total_rows <= 128

    with torch.cuda.device(hidden_states.device):
        _fused_logical_projection_kernel[grid](
            hidden_states,
            encoder_hidden_states,
            process_weight,
            processed_encoder,
            processed_hidden,
            text_rows,
            hidden_rows,
            text_seq_len,
            img_seq_len,
            hidden_states.stride(0),
            hidden_states.stride(1),
            hidden_states.stride(2),
            encoder_hidden_states.stride(0),
            encoder_hidden_states.stride(1),
            encoder_hidden_states.stride(2),
            process_weight.stride(0),
            process_weight.stride(1),
            HIDDEN_DIM=_HIDDEN_DIM,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            INPUTS_CONTIGUOUS=inputs_contiguous,
            ASYMMETRIC_EVICTION=asymmetric_eviction,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return processed_encoder, processed_hidden