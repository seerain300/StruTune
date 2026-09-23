import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_dim0_per_element_kernel(
    output_ptr,         # *bf16, shape [B, H]
    expert_ptr,         # *bf16, shape [T, H]
    indices_ptr,        # *int64, shape [T]
    B: tl.constexpr,    # batch_seq_len (rows of output)
    H: tl.constexpr,    # hidden_size (cols)
    T: tl.constexpr,    # number of expert outputs
):
    # 2D grid: one program per (row, column)
    i = tl.program_id(0)  # source row index in [0, T)
    h = tl.program_id(1)  # column index in [0, H)

    # Bounds check (generally not necessary if T and H are valid, but safe)
    if i >= T or h >= H:
        return

    # Load index (int64), then cast to int32 for address arithmetic
    idx64 = tl.load(indices_ptr + i)
    # Triton supports int64 arithmetic; cast to int32 if desired, but keep as is to avoid overflow
    # idx = idx64.to(tl.int32)  # if B < 2**31, this is fine; here B can be large, so keep int64
    idx = idx64

    # Compute pointers for the row in expert_outputs and output
    # Row-major contiguous layout: offset = i * H + h
    exp_val = tl.load(expert_ptr + i * H + h)  # bfloat16 scalar

    # Load current value from output at (idx, h), add, and store back
    out_val = tl.load(output_ptr + idx * H + h)  # bfloat16 scalar
    new_val = out_val + exp_val
    tl.store(output_ptr + idx * H + h, new_val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized replacement for:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We perform the same scatter-add using a Triton kernel.
        """
        # Ensure CUDA tensors and dtype
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Dtypes must be bfloat16."

        # Clone to match original behavior
        output = final_hidden_states.clone()

        # Ensure contiguity
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = output.shape[0]  # batch_seq_len
        H = output.shape[1]  # hidden_size
        T = token_indices.shape[0]  # number of expert outputs

        # 2D grid: one program per (row, column)
        grid = (T, H)

        # Launch Triton kernel; keep warps/stages minimal for correctness
        scatter_add_dim0_per_element_kernel[grid](
            output, expert_outputs, token_indices,
            B=B, H=H, T=T,
            num_warps=1,
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
