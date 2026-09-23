import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,    shape [N]
    M: tl.constexpr,  # total rows in out_ptr/src_ptr (batch_seq_len)
    H: tl.constexpr,  # hidden size
    N: tl.constexpr,  # total source rows (M * num_experts_per_tok)
    BLOCK_H: tl.constexpr,  # tile size across hidden dim (e.g., 256)
):
    # One program handles one source row i
    i = tl.program_id(0)
    # Guard in case grid > N (not needed if grid == N, but safe)
    if i >= N:
        return

    # Load destination row index for this source row
    dest_row = tl.load(indices_ptr + i).to(tl.int32)

    # Base offsets for source and output rows
    # out_ptr is laid out as row-major [M, H]
    # src_ptr is laid out as row-major [N, H]
    # We process hidden dimension in tiles of BLOCK_H
    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask = h_offsets < H

        # Load source values for this row i across H tile
        src_row_ptr = src_ptr + i * H
        vals = tl.load(src_row_ptr + h_offsets, mask=mask, other=0.0)  # bfloat16

        # Compute output pointer for the destination row and atomic add
        out_row_ptr = out_ptr + dest_row * H
        # Atomic add: for masked positions, 'vals' is zero so it's a no-op
        tl.atomic_add(out_row_ptr + h_offsets, vals, mask=mask)


def _run_triton_scatter_add(final_hidden_states: torch.Tensor,
                            expert_outputs: torch.Tensor,
                            token_indices: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of index_add along dim=0:
    output[token_indices[i]] += expert_outputs[i]
    with output initialized to final_hidden_states.clone().
    """
    assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
        "All tensors must be on CUDA device for Triton execution."

    # Ensure dtypes and contiguity
    out = final_hidden_states.clone()
    # Shapes
    M = out.shape[0]  # batch_seq_len
    H = out.shape[1]
    N = expert_outputs.shape[0]

    # Make sure inputs are contiguous
    out = out.contiguous()
    expert_outputs = expert_outputs.contiguous()
    token_indices = token_indices.to(torch.int32)

    # Launch configuration: one program per source row
    grid = (N,)

    # Use a fixed tile size and warps that previously gave best performance
    BLOCK_H = 256
    # Heuristic: 4 warps for 256 tile, 2 stages for pipelining
    scatter_add_rows_kernel[grid](
        out, expert_outputs, token_indices,
        M, H, N,
        BLOCK_H=BLOCK_H,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        # The Triton kernel performs the atomic scatter-add.
        # final_hidden_states is cloned and modified in-place by the kernel.
        # Ensure tensors are on CUDA; if not, we can fallback to PyTorch for correctness.
        if final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda:
            return _run_triton_scatter_add(final_hidden_states, expert_outputs, token_indices)
        else:
            # Fallback: pure PyTorch implementation for non-CUDA tensors
            out = final_hidden_states.clone()
            out.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return out


def run(*args):
    return ModelNew()(*args)
