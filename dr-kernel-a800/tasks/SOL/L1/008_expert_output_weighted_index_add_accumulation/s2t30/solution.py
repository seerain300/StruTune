import torch
import triton
import triton.language as tl


@triton.jit
def _index_add_rows_kernel(
    out_ptr,         # *const half (final_hidden_states), used for read-only if you pass a clone, but we'll write into it
    expert_ptr,      # *const half (expert_outputs)
    indices_ptr,     # *const int32 (token_indices)
    N,               # int32: number of selected tokens (rows in expert_outputs)
    H,               # int32: hidden size (columns in expert_outputs and out)
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # Each program handles one row i
    if pid >= N:
        return

    # Load the destination row index for this selected token
    idx = tl.load(indices_ptr + pid)
    # Optional: clamp idx to [0, H) if you want safety (not needed if indices are valid)
    # idx = tl.minimum(idx, H)

    # Load the entire expert vector for this row in chunks of BLOCK
    # We'll atomically add into out[idx, :]
    # Note: out_ptr + idx * H + col points to the column 'col' of row 'idx'
    # Iterate over hidden dimension in chunks
    # For robustness, keep this loop simple and masked.
    for col in range(0, H, BLOCK):
        cols = col + tl.arange(0, BLOCK)
        mask = cols < H
        # Load expert row slice
        vals = tl.load(expert_ptr + pid * H + cols, mask=mask, other=0.0)
        # Atomic add into output at row idx, columns cols
        tl.atomic_add(out_ptr + idx * H + cols, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        # Ensure contiguity and dtypes
        # Inputs are provided as bfloat16 by get_inputs; we can operate directly in bfloat16.
        # Make sure output is contiguous; we'll write into a clone to preserve original input.
        output = final_hidden_states.clone()
        # Ensure tensors are contiguous for simple pointer arithmetic
        expert_outputs = expert_outputs.contiguous()
        output = output.contiguous()  # clone already contiguous in typical setups, but enforce

        # Triton requires int32 for indexing arithmetic in kernels
        token_indices_i32 = token_indices.to(torch.int32)

        N = expert_outputs.shape[0]  # number of selected tokens
        H = expert_outputs.shape[1]  # hidden size

        # Grid: one program per row
        grid = (N,)

        # Launch kernel with tuned parameters
        # Using BLOCK=256 gives 4 chunks for H=1024, good balance. num_warps=4, num_stages=2 for decent occupancy.
        _index_add_rows_kernel[grid](
            output,            # out_ptr
            expert_outputs,    # expert_ptr
            token_indices_i32, # indices_ptr
            N,
            H,
            BLOCK=256,
            num_warps=4,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
