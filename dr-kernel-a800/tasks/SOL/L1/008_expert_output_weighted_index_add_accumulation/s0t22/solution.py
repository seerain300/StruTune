import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    output_ptr,            # *bf16, shape (M, H)
    expert_ptr,            # *bf16, shape (N, H)
    idx_ptr,               # *int32, shape (N,)
    N: tl.constexpr,       # number of source rows
    H: tl.constexpr,       # hidden size
    BLOCK_H: tl.constexpr, # tile size for hidden dimension
):
    # Each program handles one source row n
    n = tl.program_id(0)
    if n >= N:
        return

    # Load target row index (int32)
    idx = tl.load(idx_ptr + n)  # idx is int32

    # Iterate over hidden dimension in chunks
    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask = h_offsets < H

        # Load the block of hidden features for this source row
        vals = tl.load(expert_ptr + n * H + h_offsets, mask=mask, other=0.0)

        # Compute destination addresses: output[idx, h_offsets]
        dest = idx.to(tl.int64) * H + h_offsets.to(tl.int64)

        # Atomically add to output (bf16)
        tl.atomic_add(output_ptr + dest, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Performs: output = final_hidden_states.clone(); output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        with Triton kernels.
        """
        # Ensure CUDA tensors
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Dtypes must be bfloat16."
        assert token_indices.dtype in (torch.int32, torch.int64), "token_indices must be int32 or int64."

        # Clone the input buffer to match reference semantics
        output = final_hidden_states.clone()

        # Triton expects indices as int32 for arithmetic; cast safely
        idx32 = token_indices.to(torch.int32)

        M, H = output.shape
        N = expert_outputs.shape[0]

        # Kernel launch configuration
        BLOCK_H = 128  # vectorized over hidden dimension
        grid = (N,)

        # Launch Triton kernel
        scatter_add_rows_kernel[grid](
            output, expert_outputs, idx32,
            N=N, H=H,
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
