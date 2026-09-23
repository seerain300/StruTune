import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *bfloat16, shape [M, H] (writeable)
    src_ptr,          # *bfloat16, shape [N, H] (read-only)
    indices_ptr,      # *int32,    shape [N] (read-only)
    M,                # int32, total rows in output (M = batch_size * seq_len)
    H,                # int32, hidden size
    N,                # int32, number of source rows (N = M * num_experts_per_tok)
    BLOCK_H: tl.constexpr,  # tile size along hidden dimension
):
    # One program per source row i
    i = tl.program_id(0)
    if i >= N:
        return

    # Load token index for this row
    idx = tl.load(indices_ptr + i)  # int32
    if (idx < 0) or (idx >= M):
        return

    # Vector of hidden offsets for this tile
    h_offsets = tl.arange(0, BLOCK_H)

    # Iterate over hidden dimension in tiles
    for h_start in range(0, H, BLOCK_H):
        h = h_start + h_offsets  # [BLOCK_H]
        mask = h < H

        # Compute pointers for this row and tile
        out_row_ptr = out_ptr + idx * H + h
        src_row_ptr = src_ptr + i * H + h

        # Load source values (masked for tail)
        src_vals = tl.load(src_row_ptr, mask=mask, other=0.0)

        # Atomic add into output
        tl.atomic_add(out_row_ptr, src_vals, mask=mask)


def _run_triton_scatter_add(final_hidden_states: torch.Tensor,
                            expert_outputs: torch.Tensor,
                            token_indices: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of index_add along dim=0: output[token_indices[i]] += expert_outputs[i]
    """
    assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda
    assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16
    assert token_indices.dtype == torch.int32
    assert final_hidden_states.is_contiguous() and expert_outputs.is_contiguous() and token_indices.is_contiguous()

    M, H = final_hidden_states.shape
    N = expert_outputs.shape[0]

    # Output: clone input to initialize accumulation
    output = final_hidden_states.clone()

    # Grid: one program per source row
    grid = (N,)

    # Launch kernel with a balanced tile and warps
    scatter_add_rows_kernel[grid](
        output, expert_outputs, token_indices,
        M, H, N,
        BLOCK_H=256,
        num_warps=4,
        num_stages=2,
    )
    return output


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        # Ensure device consistency; evaluation provides CUDA tensors
        if not final_hidden_states.is_cuda:
            return torch.index_add(
                dim=0, index=token_indices.to(final_hidden_states.device), source=expert_outputs
            )
        # Run Triton kernel
        return _run_triton_scatter_add(final_hidden_states, expert_outputs, token_indices)


def run(*args):
    return ModelNew()(*args)
