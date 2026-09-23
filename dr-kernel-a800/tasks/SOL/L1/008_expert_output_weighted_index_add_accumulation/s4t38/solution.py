import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_by_token_index_per_elem_kernel(
    output_ptr,      # *bf16, shape [B, H]
    expert_ptr,      # *bf16, shape [T, H]
    indices_ptr,     # *int64, shape [T]
    B: tl.constexpr, # batch_seq_len (number of rows in output)
    H: tl.constexpr, # hidden_size (number of columns)
    T: tl.constexpr, # number of expert outputs
):
    # 2D grid: (i in [0, T), h in [0, H))
    i = tl.program_id(0)
    h = tl.program_id(1)

    # Bail if out of bounds (shouldn't happen because grid = (T, H))
    # Load index for this source row
    idx64 = tl.load(indices_ptr + i)
    # Triton supports int32 arithmetic; cast to int32 for address math
    idx = idx64.to(tl.int32)

    # Compute linear offsets
    # Row-major: output row-major [B, H] -> offset = idx * H + h
    out_offset = idx * H + h
    # Load current value at (idx, h)
    val = tl.load(output_ptr + out_offset)

    # Load expert value at (i, h)
    # expert_ptr is row-major [T, H] -> offset = i * H + h
    exp_offset = i * H + h
    v = tl.load(expert_ptr + exp_offset)

    # Accumulate in bfloat16
    out_val = val + v

    # Store back
    tl.store(output_ptr + out_offset, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        # Ensure all tensors are CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."
        output = final_hidden_states.clone()
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = output.shape[0]  # batch_seq_len
        H = output.shape[1]  # hidden_size
        T = token_indices.shape[0]  # number of expert outputs

        # Launch kernel: one program per (i, h)
        grid = (T, H)
        # num_warps=1 keeps each program simple; we have many programs so total parallelism is fine.
        scatter_add_by_token_index_per_elem_kernel[grid](
            output, expert_outputs, token_indices,
            B=B, H=H, T=T,
            num_warps=1,
            num_stages=1,
        )
        return output


def run(*args):
    return ModelNew()(*args)
