# solution=GPT-5.6-Sol_030_flux_concatenated_sequence_processing_with_split_triton_optimized_r24 score=1.4450940489869004 passed=True
import torch
import triton
import triton.language as tl


@triton.jit
def _projection_kernel(
    encoder_ptr,
    hidden_ptr,
    weight_ptr,
    encoder_output_ptr,
    hidden_output_ptr,
    encoder_rows,
    total_rows,
    encoder_stride_m,
    encoder_stride_k,
    hidden_stride_m,
    hidden_stride_k,
    stride_wn,
    stride_wk,
    encoder_output_stride_m,
    encoder_output_stride_n,
    hidden_output_stride_m,
    hidden_output_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    num_pid_m = tl.cdiv(total_rows, BLOCK_M)
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

    row_mask = offs_m < total_rows
    is_encoder = offs_m < encoder_rows
    hidden_m = offs_m - encoder_rows

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, 3072, BLOCK_K):
        k = k_start + offs_k

        encoder_ptrs = (
            encoder_ptr
            + offs_m[:, None] * encoder_stride_m
            + k[None, :] * encoder_stride_k
        )
        hidden_ptrs = (
            hidden_ptr
            + hidden_m[:, None] * hidden_stride_m
            + k[None, :] * hidden_stride_k
        )
        x_ptrs = tl.where(is_encoder[:, None], encoder_ptrs, hidden_ptrs)
        x = tl.load(x_ptrs, mask=row_mask[:, None], other=0.0)

        weight = tl.load(
            weight_ptr
            + offs_n[:, None] * stride_wn
            + k[None, :] * stride_wk
        )

        accumulator += tl.dot(
            x,
            tl.trans(weight),
            input_precision="tf32x3",
        )

    tl.store(
        encoder_output_ptr
        + offs_m[:, None] * encoder_output_stride_m
        + offs_n[None, :] * encoder_output_stride_n,
        accumulator,
        mask=is_encoder[:, None] & row_mask[:, None],
    )
    tl.store(
        hidden_output_ptr
        + hidden_m[:, None] * hidden_output_stride_m
        + offs_n[None, :] * hidden_output_stride_n,
        accumulator,
        mask=(~is_encoder[:, None]) & row_mask[:, None],
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
        block_k = 32
        num_warps = 4
        num_stages = 4
    elif total_rows <= 2048:
        block_m = 64
        block_n = 128
        block_k = 32
        num_warps = 4
        num_stages = 4
    else:
        block_m = 128
        block_n = 128
        block_k = 32
        num_warps = 8
        num_stages = 5

    grid = (
        triton.cdiv(total_rows, block_m) * triton.cdiv(3072, block_n),
    )

    _projection_kernel[grid](
        encoder_hidden_states,
        hidden_states,
        process_weight,
        processed_encoder,
        processed_hidden,
        encoder_rows,
        total_rows,
        encoder_hidden_states.stride(1),
        encoder_hidden_states.stride(2),
        hidden_states.stride(1),
        hidden_states.stride(2),
        process_weight.stride(0),
        process_weight.stride(1),
        processed_encoder.stride(1),
        processed_encoder.stride(2),
        processed_hidden.stride(1),
        processed_hidden.stride(2),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return processed_encoder, processed_hidden