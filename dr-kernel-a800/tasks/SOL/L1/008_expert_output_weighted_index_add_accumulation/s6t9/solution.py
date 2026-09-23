import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: copy rows from src to dst.
# src: [M, H], bf16
# dst: [M, H], bf16
@triton.jit
def _copy_rows_kernel(dst_ptr, src_ptr, M, H, BLOCK_H: tl.constexpr):
    row = tl.program_id(0)
    if row >= M:
        return
    # Iterate over the row in chunks of BLOCK_H
    for h in range(0, H, BLOCK_H):
        offs = h + tl.arange(0, BLOCK_H)
        mask = offs < H
        dst_row_ptr = dst_ptr + row * H + offs
        src_row_ptr = src_ptr + row * H + offs
        vals = tl.load(src_row_ptr, mask=mask, other=0.0)
        tl.store(dst_row_ptr, vals, mask=mask)


def _triton_copy_rows(dst: torch.Tensor, src: torch.Tensor, block_h: int = 256, num_warps: int = 4, num_stages: int = 2):
    # dst and src must be same shape [M, H], device same, dtype bf16
    assert dst.shape == src.shape
    assert dst.device == src.device
    assert dst.dtype == torch.bfloat16
    assert src.dtype == torch.bfloat16
    M, H = dst.shape
    grid = (M,)
    _copy_rows_kernel[grid](dst, src, M, H, BLOCK_H=block_h, num_warps=num_warps, num_stages=num_stages)
    # Ensure memory is flushed (optional, often not necessary for simple copy)


@torch.no_grad()
def run(
    final_hidden_states: torch.Tensor,
    expert_outputs: torch.Tensor,
    token_indices: torch.Tensor,
):
    """
    Performs the operation:
        output = final_hidden_states.clone()
        output.index_add_(dim=0, index=token_indices, source=expert_outputs)
    Triton is used for the bulk copy; PyTorch index_add is used for robust scatter-add.
    """

    # Ensure inputs are on the same device and dtype as expected
    device = final_hidden_states.device
    # Output must match final_hidden_states exactly initially
    output = torch.empty_like(final_hidden_states)  # zero-init not necessary; we'll fill from final_hidden_states via copy

    # If Triton is available and tensors are CUDA + bf16, use Triton for the copy
    use_triton = TRITON_AVAILABLE and final_hidden_states.is_cuda and final_hidden_states.dtype == torch.bfloat16

    if use_triton:
        # Triton copy: output <- final_hidden_states
        _triton_copy_rows(output, final_hidden_states, block_h=256, num_warps=4, num_stages=2)
    else:
        # Fallback: simple copy using PyTorch (fast and robust)
        output.copy_(final_hidden_states)

    # Prepare for index_add: indices must be long (int64)
    if token_indices.dtype != torch.long:
        token_indices = token_indices.to(torch.long)

    # Scatter-add along dim=0: output[row] += expert_outputs[i] for each i
    # index_add is highly optimized and robust, ensuring correctness across duplicates.
    output.index_add_(dim=0, index=token_indices, source=expert_outputs)

    return output


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        return run(final_hidden_states, expert_outputs, token_indices)


def run(*args):
    return ModelNew()(*args)
