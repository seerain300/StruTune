import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    output_ptr,            # *ptr: output tensor, shape (M, H), dtype bfloat16
    source_ptr,            # *ptr: expert_outputs tensor, shape (N, H), dtype bfloat16
    index_ptr,             # *ptr: token_indices tensor, shape (N,), dtype int32
    N,                     # int: number of source rows
    H,                     # int: hidden size
    BLOCK_H: tl.constexpr  # int: tile size across hidden dimension
):
    # 2D grid: axis 0 = row id, axis 1 = tile id across H
    row = tl.program_id(axis=0)
    tile = tl.program_id(axis=1)
    if row >= N:
        return

    # Destination index for this source row
    dest = tl.load(index_ptr + row)  # int32

    # Compute the start offset for this tile
    start = tile * BLOCK_H
    if start >= H:
        return

    # Column offsets for this tile
    offs = start + tl.arange(0, BLOCK_H)
    mask = offs < H

    # Linear pointers for source and output slices
    src_ptr = source_ptr + row * H + offs
    out_ptr = output_ptr + dest * H + offs

    # Load source values (bfloat16) with masking for out-of-range
    vals = tl.load(src_ptr, mask=mask, other=0.0)

    # Atomic add into output (vectorized across the hidden dimension)
    tl.atomic_add(out_ptr, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized version of the original run():
        Performs atomic accumulation: output[token_indices[i]] += expert_outputs[i] for all i.
        Returns the updated output tensor.
        """
        # Ensure tensors are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA device for Triton kernels."
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Allocate output (clone semantics as in the original)
        M, H = final_hidden_states.shape
        N = expert_outputs.shape[0]
        output = torch.empty_like(final_hidden_states)

        # Cast indices to int32 for Triton
        index_i32 = token_indices.to(torch.int32)

        # Choose a reasonable tile size along hidden dimension
        # 128 is a good default; for larger H, 256 may be better. Tune as needed.
        BLOCK_H = 128

        # Grid: one program per row, and tiles across H
        grid = (N, triton.cdiv(H, BLOCK_H))

        # Launch Triton kernel
        scatter_add_rows_kernel[grid](
            output, expert_outputs, index_i32,
            N, H,
            BLOCK_H=BLOCK_H,
        )

        return output


def run(*args):
    return ModelNew()(*args)
