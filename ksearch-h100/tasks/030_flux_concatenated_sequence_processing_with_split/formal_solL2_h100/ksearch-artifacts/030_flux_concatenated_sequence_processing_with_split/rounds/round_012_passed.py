# solution=GPT-5.6-Sol_030_flux_concatenated_sequence_processing_with_split_triton_optimized_r12 score=0.5389629084205072 passed=True
import torch
import triton
import triton.language as tl


@triton.jit
def _projection_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    rows,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, 3072, BLOCK_K):
        k = k_start + offs_k

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
            mask=offs_m[:, None] < rows,
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + offs_n[:, None] * stride_wn + k[None, :] * stride_wk
        )

        accumulator += tl.dot(
            x,
            tl.trans(weight),
            input_precision="tf32x3",
        )

    tl.store(
        output_ptr
        + offs_m[:, None] * stride_om
        + offs_n[None, :] * stride_on,
        accumulator,
        mask=(offs_m[:, None] < rows) & (offs_n[None, :] < 3072),
    )


def _project(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    rows = x.shape[0] * x.shape[1]

    grid = (
        triton.cdiv(rows, 64),
        triton.cdiv(3072, 128),
    )

    _projection_kernel[grid](
        x,
        weight,
        output,
        rows,
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(1),
        output.stride(1),
        output.stride(2),
        BLOCK_M=64,
        BLOCK_N=128,
        BLOCK_K=32,
        num_warps=8,
        num_stages=4,
    )
    return output


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    processed_encoder = _project(encoder_hidden_states, process_weight)
    processed_hidden = _project(hidden_states, process_weight)
    return processed_encoder, processed_hidden