import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: scatter-add expert_outputs[i, :] into output[token_indices[i], :]
# One program per index (i). Vectorize across hidden dimension in chunks of BLOCK_SIZE=256.
@triton.jit
def scatter_add_rows_kernel(
    output_ptr,         # *ptr to output [rows, hidden] (bf16)
    expert_ptr,         # *ptr to expert_outputs [num_selected_tokens, hidden] (bf16)
    indices_ptr,        # *ptr to token_indices [num_selected_tokens] (int32)
    rows,               # number of rows in output (i.e., batch_size * seq_len)
    hidden,             # hidden_size (number of columns)
    num_selected,       # num_selected_tokens
    BLOCK_SIZE: tl.constexpr,  # chunk size for hidden dimension
):
    # Each program handles one selected token (index i).
    i = tl.program_id(0)
    if i >= num_selected:
        return

    # Load the destination row index for this selected token
    row_idx = tl.load(indices_ptr + i)  # int32

    # Iterate across the hidden dimension in chunks of BLOCK_SIZE
    for start in range(0, hidden, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden

        # Compute pointers for output and expert rows
        out_ptrs = output_ptr + row_idx * hidden + offs
        exp_ptrs = expert_ptr + i * hidden + offs

        # Load expert chunk (masked for tail), then atomic add to output
        exp_vals = tl.load(exp_ptrs, mask=mask, other=0.0)
        tl.atomic_add(out_ptrs, exp_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Perform atomic accumulation of expert_outputs back to token positions.
        output[i, :] = final_hidden_states[token_indices[i], :] + expert_outputs[i, :]
        We implement the accumulation by launching a Triton kernel with atomic adds.
        """
        # Fallback to PyTorch implementation if Triton/CUDA not available
        if (not TRITON_AVAILABLE) or (final_hidden_states.device.type != "cuda"):
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices, expert_outputs)
            return output

        # Output buffer: clone final_hidden_states to start with its values
        output = final_hidden_states.clone()

        # Ensure expert_outputs contiguous and dtype matches output
        if not expert_outputs.is_contiguous():
            expert_outputs = expert_outputs.contiguous()
        if expert_outputs.dtype != output.dtype:
            expert_outputs = expert_outputs.to(output.dtype)

        # Convert token indices to int32 for Triton kernel
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)
        if not token_indices.is_contiguous():
            token_indices = token_indices.contiguous()

        rows = output.shape[0]
        hidden = output.shape[1]
        num_selected = expert_outputs.shape[0]

        # Launch Triton kernel: one program per selected token
        grid = (num_selected,)
        BLOCK_SIZE = 256  # good general choice; loops mask the tail

        scatter_add_rows_kernel[grid](
            output,                 # output_ptr
            expert_outputs,         # expert_ptr
            token_indices,          # indices_ptr
            rows,                   # rows
            hidden,                 # hidden
            num_selected,           # num_selected
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return output


def run(*args):
    return ModelNew()(*args)
