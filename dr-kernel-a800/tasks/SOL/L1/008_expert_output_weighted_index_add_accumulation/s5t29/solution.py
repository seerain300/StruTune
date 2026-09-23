import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: scatter-add expert_outputs[i, :] into output[token_indices[i], :]
# 2D grid: axis=0 over tokens, axis=1 over chunks of the hidden dimension.
@triton.jit
def scatter_add_rows_chunks_kernel(
    output_ptr,         # *ptr to output [rows, hidden]
    expert_ptr,         # *ptr to expert_outputs [num_selected_tokens, hidden]
    indices_ptr,        # *ptr to token_indices [num_selected_tokens] (int32)
    rows,               # number of rows in output (i.e., batch_size * seq_len)
    hidden,             # hidden_size (number of columns)
    num_selected,       # num_selected_tokens
    BLOCK_SIZE: tl.constexpr,  # chunk size across hidden dimension (256)
):
    pid_token = tl.program_id(axis=0)   # which token this program handles
    pid_chunk = tl.program_id(axis=1)   # which chunk along hidden this program handles

    # Each program handles exactly one token (pid_token) and one hidden chunk (pid_chunk).
    # Compute the destination row for this token.
    row_dst = tl.load(indices_ptr + pid_token)  # int32 index into output rows

    # Compute the hidden column offsets for this chunk
    start = pid_chunk * BLOCK_SIZE
    cols = start + tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden

    # Base offsets for output row and expert token
    base_out = row_dst * hidden
    base_exp = pid_token * hidden

    # Load current values from output and expert for this chunk
    out_vals = tl.load(output_ptr + base_out + cols, mask=mask, other=0)  # bf16
    exp_vals = tl.load(expert_ptr + base_exp + cols, mask=mask, other=0)  # bf16

    # Atomic add contributions for this chunk
    tl.atomic_add(output_ptr + base_out + cols, exp_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        final_hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        # Ensure Triton is available; if not, fallback to torch for correctness
        if not TRITON_AVAILABLE:
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices, expert_outputs)
            return output

        # Allocate output as a clone of final_hidden_states (dtype and device preserved)
        output = final_hidden_states.clone()

        # Ensure token_indices are int32 for Triton
        indices_i32 = token_indices.to(torch.int32)

        # Shapes
        rows = final_hidden_states.shape[0]
        hidden = final_hidden_states.shape[1]
        num_selected = expert_outputs.shape[0]

        # Launch Triton kernel: 2D grid over tokens and hidden chunks
        # BLOCK_SIZE=256 works well across a range of hidden sizes.
        BLOCK_SIZE = 256
        grid = (num_selected, triton.cdiv(hidden, BLOCK_SIZE))

        scatter_add_rows_chunks_kernel[grid](
            output,
            expert_outputs,
            indices_i32,
            rows,
            hidden,
            num_selected,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
