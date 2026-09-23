import torch
import triton
import triton.language as tl


@triton.jit
def copy_rows_kernel(output_ptr, src_ptr, B: tl.int32, H: tl.int32):
    # Grid: (B, H). Each program copies one element: output[i, h] = src[i, h]
    pid_row = tl.program_id(0)  # i in [0, B)
    pid_col = tl.program_id(1)  # h in [0, H)

    if pid_row >= B:
        return

    # Flat index for source and destination
    index = pid_row * H + pid_col

    # Load/store as bfloat16
    val = tl.load(src_ptr + index, dtype=tl.bfloat16)
    tl.store(output_ptr + index, val)


@triton.jit
def scatter_add_rows_atomic_kernel(output_ptr, expert_ptr, indices_ptr, B: tl.int32, H: tl.int32, T: tl.int32):
    # Grid: (T, H). Each program handles one source row i and one column h.
    pid_i = tl.program_id(0)  # i in [0, T)
    pid_h = tl.program_id(1)  # h in [0, H)

    if pid_i >= T:
        return

    # Load index and cast to int32 for pointer arithmetic
    idx64 = tl.load(indices_ptr + pid_i)
    idx = idx64.to(tl.int32)

    # Bounds check (indices are valid by construction, but keep safety)
    if idx < 0 or idx >= B:
        return

    # Compute flat indices for output and expert
    out_index = idx * H + pid_h
    expert_index = pid_i * H + pid_h

    # Load bf16 values and atomic add
    val_out = tl.load(output_ptr + out_index, dtype=tl.bfloat16)
    val_exp = tl.load(expert_ptr + expert_index, dtype=tl.bfloat16)
    tl.atomic_add(output_ptr + out_index, val_exp)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        All computation is done by Triton kernels.
        """
        # 断言并准备
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."
        B = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        T = expert_outputs.shape[0]

        # Output starts empty; we will perform the clone and scatter-add in Triton
        output = torch.empty_like(final_hidden_states)

        # Ensure contiguous memory for predictable pointer arithmetic
        final_hidden_states = final_hidden_states.contiguous()
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # 1) Copy final_hidden_states into output using Triton
        # Grid: (B, H)
        grid_copy = (B, H)
        copy_rows_kernel[grid_copy](
            output, final_hidden_states, B, H,
            num_warps=1,
            num_stages=1,
        )

        # 2) Scatter-add: for each i in [0, T) and h in [0, H), add expert_outputs[i, h] to output[token_indices[i], h]
        # Grid: (T, H)
        grid_add = (T, H)
        scatter_add_rows_atomic_kernel[grid_add](
            output, expert_outputs, token_indices, B, H, T,
            num_warps=1,
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
