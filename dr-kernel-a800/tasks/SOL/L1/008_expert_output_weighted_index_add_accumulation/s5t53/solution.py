import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: scatter-add expert_outputs[i, :] into output[token_indices[i], :]
# One program per selected token i. Vectorize across hidden dimension in chunks of BLOCK_SIZE=256.
@triton.jit
def scatter_add_per_index_kernel(
    output_ptr,         # *ptr to output [rows, hidden]
    expert_ptr,         # *ptr to expert_outputs [num_selected_tokens, hidden]
    indices_ptr,        # *ptr to token_indices [num_selected_tokens] (int32)
    rows,               # number of rows in output (i.e., batch_size * seq_len)
    hidden,             # hidden_size (number of columns)
    num_selected,       # num_selected_tokens
    BLOCK_SIZE: tl.constexpr,
):
    # program id: each program handles one selected token i
    pid = tl.program_id(0)
    if pid >= num_selected:
        return

    # Load target row index for this token (int32)
    row_idx = tl.load(indices_ptr + pid)
    if row_idx < 0 or row_idx >= rows:
        return

    # Vectorize across hidden dimension
    for start in range(0, hidden, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden

        # Load the corresponding hidden slice from the expert row
        expert_vals = tl.load(expert_ptr + pid * hidden + offs, mask=mask, other=0.0)
        # Compute output pointer for the selected row
        out_ptrs = output_ptr + row_idx * hidden + offs
        # Atomic add the slice
        tl.atomic_add(out_ptrs, expert_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Perform atomic accumulation of expert_outputs back to token positions.
        - final_hidden_states: accumulation buffer (M, H), dtype bfloat16
        - expert_outputs: weighted outputs (N_selected, H), dtype bfloat16
        - token_indices: original token positions (N_selected,), dtype long
        Returns updated final_hidden_states with expert contributions added.
        """
        # Ensure tensors are on CUDA and have the expected dtype
        assert TRITON_AVAILABLE, "Triton is not available"
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Expect bfloat16 tensors."

        # Clone to avoid modifying input
        output = final_hidden_states.clone()

        # Prepare indices as int32 for Triton
        indices_i32 = token_indices.to(torch.int32)

        # Shapes
        M = output.shape[0]
        H = output.shape[1]
        N = expert_outputs.shape[0]

        # Grid: one program per selected token
        grid = (N,)

        # Use BLOCK_SIZE=256 for robust performance across a wide range of H
        BLOCK_SIZE = 256

        # Launch kernel; 4 warps is a balanced default for simple atomic kernels
        scatter_add_per_index_kernel[grid](
            output,              # output_ptr
            expert_outputs,      # expert_ptr
            indices_i32,         # indices_ptr (int32)
            M,                   # rows
            H,                   # hidden
            N,                   # num_selected_tokens
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
