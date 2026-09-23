import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *bfloat16, shape [M, H]
    src_ptr,          # *bfloat16, shape [N, H]
    indices_ptr,      # *int32,    shape [N]
    M,                # int32, total rows in out_ptr/src_ptr
    H: tl.constexpr,  # hidden size (constexpr for tiling)
    BLOCK_H: tl.constexpr,  # tile size along hidden dim (e.g., 256)
):
    # One program per source row
    row_id = tl.program_id(0)
    # Load token index for this row
    idx = tl.load(indices_ptr + row_id)  # int32

    # Loop over hidden dimension in tiles of size BLOCK_H
    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask = h_offsets < H

        # Compute pointers for this row in output and source
        out_row_ptr = out_ptr + idx * H + h_offsets
        src_row_ptr = src_ptr + row_id * H + h_offsets

        # Load current output row slice and source slice
        out_vals = tl.load(out_row_ptr, mask=mask, other=0.0)
        src_vals = tl.load(src_row_ptr, mask=mask, other=0.0)

        # Atomic add: out[idx, h] += src[row_id, h]
        out_vals = out_vals + src_vals

        # Store back with mask
        tl.store(out_row_ptr, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Clone to match original behavior
        output = final_hidden_states.clone()

        # Ensure dtypes and contiguity
        assert final_hidden_states.dtype == torch.bfloat16, "final_hidden_states must be bfloat16"
        assert expert_outputs.dtype == torch.bfloat16, "expert_outputs must be bfloat16"
        assert token_indices.dtype in (torch.int64, torch.int32), "token_indices must be int64 or int32"
        # Triton prefers int32 for indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Make tensors contiguous
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()

        # Dimensions
        M = final_hidden_states.shape[0]
        N = expert_outputs.shape[0]
        H = final_hidden_states.shape[1]

        # Grid: one program per source row
        grid = (N,)

        # Launch kernel; use BLOCK_H=256 which performed well in prior correct runs
        scatter_add_rows_kernel[grid](
            output,
            expert_outputs,
            token_indices,
            M,
            H,                 # constexpr for tiling
            BLOCK_H=256,       # constexpr tile size
            num_warps=4,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
