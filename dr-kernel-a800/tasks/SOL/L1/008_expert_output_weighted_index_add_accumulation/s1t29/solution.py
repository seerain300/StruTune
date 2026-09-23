import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *bfloat16, shape [M, H], contiguous
    src_ptr,          # *bfloat16, shape [N, H], contiguous
    indices_ptr,      # *int32,    shape [N]
    M,                # int (runtime)
    stride_out_row,   # int (elements)
    stride_out_col,   # int (elements)
    stride_src_row,   # int (elements)
    stride_src_col,   # int (elements)
    H: tl.constexpr,  # hidden size, compile-time constant for unrolling
    BLOCK_H: tl.constexpr,  # tile size across hidden dim
):
    # One program per source row
    row_id = tl.program_id(0)
    # Guard: if row_id >= N (grid may be larger), do nothing
    if row_id >= N:
        return

    # Load the destination row index
    dest_row = tl.load(indices_ptr + row_id)  # int32
    # Compute base pointers for this row
    out_row_base = out_ptr + dest_row * stride_out_row
    src_row_base = src_ptr + row_id * stride_src_row

    # Vector of hidden offsets
    h_offsets = tl.arange(0, BLOCK_H)

    # Iterate across the hidden dimension in tiles of size BLOCK_H
    for h_start in range(0, H, BLOCK_H):
        h_idx = h_start + h_offsets
        mask = h_idx < H

        # Load source values for this row and tile
        src_vals = tl.load(src_row_base + h_idx * stride_src_col, mask=mask, other=0.0)

        # Atomic add into destination row
        tl.atomic_add(out_row_base + h_idx * stride_out_col, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized forward that performs:
          output = torch.zeros_like(final_hidden_states)
          output.index_add_(dim=0, token_indices, expert_outputs)
        Implemented via Triton kernel with atomic scatter-add.
        """
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA device"
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "dtype must be bfloat16"
        assert token_indices.dtype in (torch.int32, torch.int64), "token_indices must be int32 or int64"

        # Make sure tensors are contiguous
        out = torch.zeros(final_hidden_states.shape, dtype=final_hidden_states.dtype, device=final_hidden_states.device)
        expert_outputs = expert_outputs.contiguous()
        final_hidden_states = final_hidden_states.contiguous()

        # Cast indices to int32 for Triton
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        M = out.shape[0]
        H = out.shape[1]
        N = expert_outputs.shape[0]

        # Launch one program per source row
        grid = (N,)

        # Select BLOCK_H; 256 is a robust choice for typical H up to 1024
        BLOCK_H = 256

        scatter_add_rows_kernel[grid](
            out, expert_outputs, token_indices,
            M,
            out.stride(0), out.stride(1),
            expert_outputs.stride(0), expert_outputs.stride(1),
            H=H,
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
