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
    output_ptr,         # *ptr to output [rows, hidden]
    expert_ptr,         # *ptr to expert_outputs [num_selected_tokens, hidden]
    indices_ptr,        # *ptr to token_indices [num_selected_tokens] (int32)
    rows,               # number of rows in output (i.e., batch_size * seq_len)
    hidden,             # hidden_size (number of columns)
    num_selected,       # num_selected_tokens
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one selected index (row in output)
    pid = tl.program_id(axis=0)
    if pid >= num_selected:
        return

    # Compute the target output row for this selected index
    row = tl.load(indices_ptr + pid)
    if row < 0 or row >= rows:
        # Defensive: skip if out-of-range, though token_indices should be valid.
        return

    # Vectorize across hidden dimension in chunks of BLOCK_SIZE
    for k in range(0, hidden, BLOCK_SIZE):
        offs = tl.arange(0, BLOCK_SIZE)
        hid = k + offs
        mask = hid < hidden

        # Load source expert chunk (masked for tail)
        src_ptr = expert_ptr + pid * hidden + k + offs
        vals = tl.load(src_ptr, mask=mask, other=0.0)

        # Load destination row chunk
        dst_ptr = output_ptr + row * hidden + k + offs
        out = tl.load(dst_ptr, mask=mask, other=0.0)

        # Atomic add into destination
        tl.atomic_add(dst_ptr, vals, mask=mask)


def _run_triton(final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
    # Clone to avoid modifying input in-place (matches original behavior)
    output = final_hidden_states.clone()
    # Ensure dtypes/devices are consistent; Triton prefers int32 for indices
    token_indices_i32 = token_indices.to(torch.int32)

    # Ensure tensors are contiguous
    output = output.contiguous()
    expert_outputs = expert_outputs.contiguous()
    token_indices_i32 = token_indices_i32.contiguous()

    rows = output.shape[0]
    hidden = output.shape[1]
    num_selected = expert_outputs.shape[0]

    # Launch Triton kernel: one program per selected token
    grid = (num_selected,)
    scatter_add_rows_kernel[grid](
        output, expert_outputs, token_indices_i32,
        rows, hidden, num_selected,
        BLOCK_SIZE=256,
        num_warps=4,  # modest warp count for good occupancy across shapes
        num_stages=2,
    )
    return output


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Triton-only computation: host code performs minimal data prep and launch.
        return _run_triton(final_hidden_states, expert_outputs, token_indices)


def run(*args):
    return ModelNew()(*args)
