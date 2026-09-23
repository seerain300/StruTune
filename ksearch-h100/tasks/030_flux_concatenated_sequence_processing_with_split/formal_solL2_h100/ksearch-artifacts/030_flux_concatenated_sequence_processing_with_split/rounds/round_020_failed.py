# solution=GPT-5.6-Sol_030_flux_concatenated_sequence_processing_with_split_triton_optimized_r20 score=-1.0 passed=False
import torch
import triton
import triton.language as tl


@triton.jit
def _projection_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    rows,
    input_stride_m,
    input_stride_k,
    stride_wn,
    stride_wk,
    output_stride_m,
    output_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    num_pid_m = tl.cdiv(rows, BLOCK_M)
    num_pid_n = tl.cdiv(3072, BLOCK_N)

    pid = tl.program_id(0)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)

    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + pid_in_group % group_size_m
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    row_mask = offs_m < rows
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, 3072, BLOCK_K):
        k = k_start + offs_k

        x = tl.load(
            input_ptr
            + offs_m[:, None] * input_stride_m
            + k[None, :] * input_stride_k,
            mask=row_mask[:, None],
            other=0.0,
        )

        weight = tl.load(
            weight_ptr
            + offs_n[:, None] * stride_wn
            + k[None, :] * stride_wk,
        )

        accumulator += tl.dot(
            x,
            tl.trans(weight),
            input_precision="tf32x3",
        )

    tl.store(
        output_ptr
        + offs_m[:, None] * output_stride_m
        + offs_n[None, :] * output_stride_n,
        accumulator,
        mask=row_mask[:, None],
    )


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    processed_encoder = torch.empty_like(encoder_hidden_states)
    processed_hidden = torch.empty_like(hidden_states)

    encoder_rows = (
        encoder_hidden_states.shape[0] * encoder_hidden_states.shape[1]
    )
    hidden_rows = hidden_states.shape[0] * hidden_states.shape[1]
    total_rows = encoder_rows + hidden_rows

    if total_rows <= 512:
        block_m = 32
        block_n = 128
        num_warps = 4
        num_stages = 3

        grid = (
            triton.cdiv(total_rows, block_m) * triton.cdiv(3072, block_n),
        )

        _projection_kernel[grid](
            encoder_hidden_states,
            process_weight,
            processed_encoder,
            encoder_rows,
            encoder_hidden_states.stride(1),
            encoder_hidden_states.stride(2),
            process_weight.stride(0),
            process_weight.stride(1),
            processed_encoder.stride(1),
            processed_encoder.stride(2),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=32,
            GROUP_M=8,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return processed_encoder, processed_hidden

    if encoder_rows <= 512:
        encoder_block_m = 32
        encoder_warps = 4
        encoder_stages = 3
    elif encoder_rows <= 2048:
        encoder_block_m = 64
        encoder_warps = 4
        encoder_stages = 4
    else:
        encoder_block_m = 128
        encoder_warps = 8
        encoder_stages = 5

    if hidden_rows <= 512:
        hidden_block_m = 32
        hidden_warps = 4
        hidden_stages = 3
    elif hidden_rows <= 2048:
        hidden_block_m = 64
        hidden_warps = 4
        hidden_stages = 4
    else:
        hidden_block_m = 128
        hidden_warps = 8
        hidden_stages = 5

    block_n = 128

    encoder_grid = (
        triton.cdiv(encoder_rows, encoder_block_m)
        * triton.cdiv(3072, block_n),
    )
    hidden_grid = (
        triton.cdiv(hidden_rows, hidden_block_m)
        * triton.cdiv(3072, block_n),
    )

    _projection_kernel[encoder_grid](
        encoder_hidden_states,
        process_weight,
        processed_encoder,
        encoder_rows,
        encoder_hidden_states.stride(1),
        encoder_hidden_states.stride(2),
        process_weight.stride(0),
        process_weight.stride(1),
        processed_encoder.stride(1),
        processed_encoder.stride(2),
        BLOCK_M=encoder_block_m,
        BLOCK_N=block_n,
        BLOCK_K=32,
        GROUP_M=8,
        num_warps=encoder_warps,
        num_stages=encoder_stages,
    )

    _projection_kernel[hidden_grid](
        hidden_states,
        process_weight,
        processed_hidden,
        hidden_rows,
        hidden_states.stride(1),
        hidden_states.stride(2),
        process_weight.stride(0),
        process_weight.stride(1),
        processed_hidden.stride(1),
        processed_hidden.stride(2),
        BLOCK_M=hidden_block_m,
        BLOCK_N=block_n,
        BLOCK_K=32,
        GROUP_M=8,
        num_warps=hidden_warps,
        num_stages=hidden_stages,
    )

    return processed_encoder, processed_hidden