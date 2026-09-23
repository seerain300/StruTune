import torch
import triton
import triton.language as tl


@triton.jit
def scatter_copy_and_add_rows_kernel(
    out_ptr,          # *bfloat16, shape [M, H]
    final_ptr,        # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,    shape [N]
    M: tl.constexpr,  # number of rows in final/out
    H: tl.constexpr,  # hidden size
    N,                # number of source rows
    out_stride0: tl.constexpr,  # stride of out along dim 0 (elements)
    out_stride1: tl.constexpr,  # stride of out along dim 1 (elements)
    final_stride0: tl.constexpr,  # stride of final along dim 0 (elements)
    final_stride1: tl.constexpr,  # stride of final along dim 1 (elements)
    src_stride0: tl.constexpr,  # stride of src along dim 0 (elements)
    src_stride1: tl.constexpr,  # stride of src along dim 1 (elements)
    BLOCK_H: tl.constexpr,      # tile size along H
):
    # One program per source row
    row = tl.program_id(0)  # in [0, N)

    # Destination row index in output
    dest_row = tl.load(indices_ptr + row)  # int32

    # Process hidden dimension in tiles
    for off in range(0, H, BLOCK_H):
        h_offsets = off + tl.arange(0, BLOCK_H)
        mask = h_offsets < H

        # Copy final_hidden_states slice at dest_row into output
        final_row_ptr = final_ptr + dest_row * final_stride0
        vals = tl.load(final_row_ptr + h_offsets * final_stride1, mask=mask, other=0.0)
        out_row_ptr = out_ptr + dest_row * out_stride0
        tl.store(out_row_ptr + h_offsets * out_stride1, vals, mask=mask)

        # Atomic add expert_outputs slice into output at dest_row
        src_row_ptr = src_ptr + row * src_stride0
        src_vals = tl.load(src_row_ptr + h_offsets * src_stride1, mask=mask, other=0.0)
        tl.atomic_add(out_row_ptr + h_offsets * out_stride1, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        """
        Triton-optimized implementation of:
          output = final_hidden_states.clone()
          for i in range(N): output[token_indices[i]] += expert_outputs[i]
        The entire computation is done by Triton; no torch elementwise/reduction on device tensors.
        """
        # Ensure CUDA tensors and contiguity
        assert final_hidden_states.is_cuda, "final_hidden_states must be on CUDA"
        assert expert_outputs.is_cuda, "expert_outputs must be on CUDA"
        assert token_indices.is_cuda, "token_indices must be on CUDA"

        M, H = final_hidden_states.shape
        N = expert_outputs.shape[0]

        # Make source contiguous for predictable strides
        src = expert_outputs.contiguous()

        # Prepare output tensor (we will fill it via Triton kernel)
        output = torch.empty((M, H), dtype=final_hidden_states.dtype, device=final_hidden_states.device)

        # Triton prefers int32 indices
        idx32 = token_indices.to(torch.int32)

        # Launch Triton kernel: one program per source row
        BLOCK_H = 256  # tile size along H; good balance for typical H up to 1024
        grid = (N,)

        scatter_copy_and_add_rows_kernel[grid](
            output, final_hidden_states, src, idx32,
            M, H, N,
            output.stride(0), output.stride(1),
            final_hidden_states.stride(0), final_hidden_states.stride(1),
            src.stride(0), src.stride(1),
            BLOCK_H,
            num_warps=4,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
