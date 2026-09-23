import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_serial_kernel(
    output_ptr,          # *const bfloat16
    expert_ptr,          # *const bfloat16
    indices_ptr,         # *const int64
    B: tl.constexpr,     # int: number of rows in output (batch_seq_len)
    H: tl.constexpr,     # int: number of columns (hidden_size)
    T: tl.constexpr,     # int: number of source rows (num_selected_tokens)
):
    # One program per source row
    i = tl.program_id(0)
    # Bounds check: if i >= T, do nothing (defensive, though grid ensures i < T)
    # Note: Triton grid is typically set to exactly T, so this is not usually needed.
    if i >= T:
        return

    # Load token index for this row i (int64), then convert to int32 for pointer arithmetic
    idx64 = tl.load(indices_ptr + i)
    # Cast to int32. B is the number of rows and is typically within int32 range here.
    idx = idx64.to(tl.int32)

    # Iterate over each hidden dimension h
    # Serial processing to minimize rounding-order differences
    for h in range(0, H):
        # Load the expert value for this row i and column h (bf16)
        # Memory layout is row-major: row i has base offset i * H + h
        e = tl.load(expert_ptr + i * H + h)

        # Load current output at row idx and column h (bf16)
        out_ptr = output_ptr + idx * H + h
        curr = tl.load(out_ptr)

        # Accumulate (bf16 arithmetic). Triton will perform bf16 math here.
        new_val = curr + e

        # Store back
        tl.store(out_ptr, new_val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-only implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        This kernel performs the scatter-add row-wise in Triton, iterating serially over hidden columns
        to minimize rounding-order differences and ensure correctness.
        """
        # Ensure tensors are on CUDA for Triton
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA for Triton."

        # Clone to match original behavior (index_add writes into a separate output)
        output = final_hidden_states.clone()

        # Ensure contiguity for pointer arithmetic
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = output.shape[0]  # batch_seq_len
        H = output.shape[1]  # hidden_size
        T = token_indices.shape[0]  # num_selected_tokens

        # Launch Triton kernel: one program per source row
        grid = (T,)

        scatter_add_rows_serial_kernel[grid](
            output, expert_outputs, token_indices,
            B=B, H=H, T=T,
            num_warps=1,
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
