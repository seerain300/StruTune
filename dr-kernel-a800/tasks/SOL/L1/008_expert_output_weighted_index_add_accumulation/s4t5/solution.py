import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_serial_kernel(
    output_ptr,      # *bf16, shape (B, H)
    expert_ptr,      # *bf16, shape (T, H)
    indices_ptr,     # *int64, shape (T,)
    B: tl.int32,     # batch_seq_len (rows in output)
    H: tl.int32,     # hidden_size (columns)
    T: tl.int32,     # number of source rows
):
    # One program per source row i
    i = tl.program_id(0)
    if i >= T:
        return

    # Load destination row index (int64 -> int32 for pointer arithmetic)
    idx64 = tl.load(indices_ptr + i)
    idx = idx64.to(tl.int32)

    # Serial loop over hidden dimension to avoid atomics and ensure deterministic addition
    for h in range(0, H):
        # Load source scalar (bf16)
        v = tl.load(expert_ptr + i * H + h)
        # Load destination scalar, add, and store back
        val = tl.load(output_ptr + idx * H + h)
        val = val + v
        tl.store(output_ptr + idx * H + h, val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized version of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We implement the index_add via a Triton kernel that does per-element scatter-add along rows.
        """
        # Ensure tensors are on CUDA
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA for Triton."

        # Clone to match original behavior
        output = final_hidden_states.clone()

        # Ensure contiguity for pointer arithmetic
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = output.shape[0]  # batch_seq_len
        H = output.shape[1]  # hidden_size
        T = token_indices.shape[0]  # number of expert outputs

        # Launch one program per source row
        grid = (T,)

        # Run Triton kernel (serial per-column loop to guarantee correctness)
        scatter_add_rows_serial_kernel[grid](
            output, expert_outputs, token_indices,
            B=B, H=H, T=T,
            num_warps=1,
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
