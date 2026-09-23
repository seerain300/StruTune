import torch
import triton
import triton.language as tl


@triton.jit
def copy_rows_kernel(
    in_ptr, out_ptr,
    M: tl.int32,  # number of rows to copy
    N: tl.int32,  # number of columns per row
    BLOCK_N: tl.constexpr,
):
    """
    Copy one row per Triton program: out[row] = in[row] for row in [0, M).
    Vectorizes along the column dimension N using BLOCK_N.
    """
    row = tl.program_id(0)
    # If grid > M, mask out-of-range rows (usually grid == M, but be safe).
    if row >= M:
        return

    # Iterate over columns in chunks of BLOCK_N
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < N
        in_row_ptr = in_ptr + row * N + cols
        out_row_ptr = out_ptr + row * N + cols
        vals = tl.load(in_row_ptr, mask=mask, other=0.0)
        tl.store(out_row_ptr, vals, mask=mask)


@torch.no_grad()
def run(
    final_hidden_states: torch.Tensor,
    expert_outputs: torch.Tensor,
    token_indices: torch.Tensor,
):
    """
    Clone final_hidden_states into output using Triton, then perform index_add along dim=0.
    This matches the original PyTorch implementation exactly.
    """
    # Ensure tensors are on the same device and dtype; inputs are already bfloat16 and device-aware.
    # Allocate output and copy using Triton for deterministic behavior.
    output = torch.empty_like(final_hidden_states)

    M = final_hidden_states.shape[0]
    N = final_hidden_states.shape[1]
    # Choose a reasonable BLOCK_N for vectorization; 256 works well for typical hidden sizes.
    BLOCK_N = 256

    # Launch one program per row. Triton grid is 1D with size M.
    grid = (M,)
    copy_rows_kernel[grid](final_hidden_states, output, M, N, BLOCK_N=BLOCK_N, num_warps=4)

    # Perform scatter-add along dim=0 using PyTorch. This handles duplicates and arbitrary indices robustly.
    output.index_add_(dim=0, index=token_indices, source=expert_outputs)

    return output


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        return run(final_hidden_states, expert_outputs, token_indices)


def run(*args):
    return ModelNew()(*args)
