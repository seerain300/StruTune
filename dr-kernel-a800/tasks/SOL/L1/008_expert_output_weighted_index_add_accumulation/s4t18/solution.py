import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_dim0_per_element_kernel(
    output_ptr,         # *bf16, shape [B, H]
    expert_ptr,         # *bf16, shape [T, H]
    indices_ptr,        # *int64, shape [T]
    B: tl.constexpr,    # number of rows in output (batch_seq_len)
    H: tl.constexpr,    # number of columns (hidden_size)
    T: tl.constexpr,    # number of expert outputs
):
    # 2D grid: one program per (row i, column h)
    row = tl.program_id(0)  # i in [0, T)
    col = tl.program_id(1)  # h in [0, H)

    # Defensive bounds (grid is set to (T, H), so row < T and col < H)
    # We still guard in case of over-launch.
    if row >= T or col >= H:
        return

    # Load token index for this row (int64), cast to int32 for pointer arithmetic
    idx64 = tl.load(indices_ptr + row)
    idx = idx64.to(tl.int32)

    # Load the value from expert_outputs[row, col] (bf16)
    v = tl.load(expert_ptr + row * H + col)

    # Load current value from output[idx, col] (bf16)
    current = tl.load(output_ptr + idx * H + col)

    # Compute and store the result: output[idx, col] = current + v
    out = current + v
    tl.store(output_ptr + idx * H + col, out)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized version of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We perform the same scatter-add using a Triton kernel.
        """
        # Ensure tensors are on CUDA for Triton
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."

        # Clone to match original behavior (index_add writes into a separate output)
        output = final_hidden_states.clone()

        # Make tensors contiguous for performance
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = output.shape[0]  # batch_seq_len (number of rows to scatter into)
        H = output.shape[1]  # hidden_size (number of columns)
        T = token_indices.shape[0]  # number of expert outputs to scatter

        # Launch 2D grid: one program per (row, column)
        grid = (T, H)

        # Run Triton kernel. Use modest num_warps; each program is light (per-element).
        scatter_add_dim0_per_element_kernel[grid](
            output, expert_outputs, token_indices,
            B=B, H=H, T=T,
            num_warps=1,
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
