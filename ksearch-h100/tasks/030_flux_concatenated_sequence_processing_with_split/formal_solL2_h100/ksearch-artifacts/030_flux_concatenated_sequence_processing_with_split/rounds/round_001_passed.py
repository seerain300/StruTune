# solution=GPT-5.6-Sol_030_flux_concatenated_sequence_processing_with_split_triton_optimized_r1 score=1.5215433270418957 passed=True
import torch
import triton
import triton.language as tl


@triton.jit
def _unified_projection_kernel(
    hidden_states,
    encoder_hidden_states,
    process_weight,
    processed_encoder,
    processed_hidden,
    total_rows: tl.constexpr,
    encoder_rows: tl.constexpr,
    stride_wo,
    stride_wi,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(total_rows, BLOCK_M)
    num_pid_n = tl.cdiv(HIDDEN_DIM, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + pid_in_group % group_size_m
    pid_n = pid_in_group // group_size_m

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < total_rows

    is_encoder = rows < encoder_rows
    stream_rows = tl.where(is_encoder, rows, rows - encoder_rows)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, HIDDEN_DIM, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)

        encoder_ptrs = (
            encoder_hidden_states
            + rows[:, None] * HIDDEN_DIM
            + k[None, :]
        )
        hidden_ptrs = (
            hidden_states
            + stream_rows[:, None] * HIDDEN_DIM
            + k[None, :]
        )
        input_ptrs = tl.where(
            is_encoder[:, None],
            encoder_ptrs,
            hidden_ptrs,
        )

        weight_ptrs = (
            process_weight
            + cols[None, :] * stride_wo
            + k[:, None] * stride_wi
        )

        values = tl.load(
            input_ptrs,
            mask=row_mask[:, None],
            other=0.0,
        )
        weights = tl.load(weight_ptrs)

        accumulator += tl.dot(
            values,
            weights,
            input_precision="tf32x3",
        )

    encoder_output_ptrs = (
        processed_encoder
        + rows[:, None] * HIDDEN_DIM
        + cols[None, :]
    )
    hidden_output_ptrs = (
        processed_hidden
        + stream_rows[:, None] * HIDDEN_DIM
        + cols[None, :]
    )
    output_ptrs = tl.where(
        is_encoder[:, None],
        encoder_output_ptrs,
        hidden_output_ptrs,
    )

    tl.store(
        output_ptrs,
        accumulator,
        mask=row_mask[:, None],
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
    hidden_dim = hidden_states.shape[2]

    processed_encoder = torch.empty_like(encoder_hidden_states)
    processed_hidden = torch.empty_like(hidden_states)

    encoder_rows = batch_size * text_seq_len
    total_rows = encoder_rows + batch_size * img_seq_len

    if total_rows <= 512:
        block_m = 64
        block_n = 64
        block_k = 32
        group_m = 8
        num_row_groups = triton.cdiv(
            triton.cdiv(total_rows, block_m),
            group_m,
        )
        num_warps = 8 if num_row_groups == 1 else 4
        num_stages = 5
    else:
        block_m = 128
        block_n = 128
        block_k = 64
        group_m = 16
        num_row_groups = triton.cdiv(
            triton.cdiv(total_rows, block_m),
            group_m,
        )
        num_warps = 8 if num_row_groups <= 4 else 4
        num_stages = 3

    grid = (
        triton.cdiv(total_rows, block_m)
        * triton.cdiv(hidden_dim, block_n),
    )

    _unified_projection_kernel[grid](
        hidden_states,
        encoder_hidden_states,
        process_weight,
        processed_encoder,
        processed_hidden,
        total_rows,
        encoder_rows,
        process_weight.stride(0),
        process_weight.stride(1),
        HIDDEN_DIM=hidden_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return processed_encoder, processed_hidden