import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_dim0_kernel(
    output_ptr,         # *bf16, shape [B, H]
    expert_ptr,         # *bf16, shape [T, H]
    indices_ptr,        # *int64, shape [T]
    B: tl.constexpr,    # batch_seq_len (rows of output)
    H: tl.constexpr,    # hidden_size (cols)
    T: tl.constexpr,    # number of expert outputs
):
    # 2D grid: one program per (row i, column h)
    i = tl.program_id(0)  # row index in [0, T)
    h = tl.program_id(1)  # column index in [0, H)

    # Bounds check: although i < T and h < H by construction, keep for safety
    # Triton grid ensures i < T, h < H, but masking helps in general cases.
    if (i >= T) or (h >= H):
        return

    # Load token index for this row (int64), then convert to int32 for pointer arithmetic
    idx64 = tl.load(indices_ptr + i)
    idx = idx64.to(tl.int32)

    # Load the corresponding value from expert_outputs at column h
    v = tl.load(expert_ptr + i * H + h)  # v is bfloat16

    # Compute output address and add v to that position
    out_ptr = output_ptr + idx * H + h
    tl.store(out_ptr, tl.load(out_ptr) + v)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        # Ensure tensors are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."

        # Clone to match original behavior
        output = final_hidden_states.clone()
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = output.shape[0]  # batch_seq_len
        H = output.shape[1]  # hidden_size
        T = token_indices.shape[0]  # number of expert outputs

        # Launch Triton kernel: one program per (row, column)
        grid = (T, H)

        scatter_add_dim0_kernel[grid](
            output, expert_outputs, token_indices,
            B=B, H=H, T=T,
            num_warps=1,
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
