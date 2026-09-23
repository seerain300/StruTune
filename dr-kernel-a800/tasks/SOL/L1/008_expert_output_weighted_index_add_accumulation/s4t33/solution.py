import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_cols_kernel(
    output_ptr,         # *bfloat16, shape (B, H)
    expert_ptr,         # *bfloat16, shape (T, H)
    indices_ptr,        # *int64,    shape (T,)
    B: tl.int32,        # batch_seq_len
    H: tl.int32,        # hidden_size
    T: tl.int32,        # num_selected_tokens
):
    # 2D grid: program_id(0) = i in [0, T), program_id(1) = h in [0, H)
    i = tl.program_id(0)
    h = tl.program_id(1)

    # Guard if grid is larger than T or H (shouldn't happen if we set grid exactly to (T, H))
    if i >= T or h >= H:
        return

    # Load token index (int64), then cast to int32 for address arithmetic
    idx64 = tl.load(indices_ptr + i)
    idx = idx64.to(tl.int32)

    # Compute pointers
    # Output row pointer for this idx and column h
    out_row_ptr = output_ptr + idx * H + h
    # Expert row pointer for this i and column h
    exp_row_ptr = expert_ptr + i * H + h

    # Load value from expert_outputs[i, h] as bfloat16
    v = tl.load(exp_row_ptr)  # dtype inferred from tensor (bf16)

    # Load current output value and add
    curr = tl.load(out_row_ptr)
    new_v = curr + v

    # Store back (Triton will handle bf16 store)
    tl.store(out_row_ptr, new_v)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We perform the scatter-add explicitly in a Triton kernel, avoiding PyTorch ops on tensors.
        """
        # Ensure CUDA tensors and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All inputs must be CUDA tensors for Triton execution."
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Clone to match original behavior
        output = final_hidden_states.clone()

        # Shapes
        B = output.shape[0]  # batch_seq_len
        H = output.shape[1]  # hidden_size
        T = token_indices.shape[0]  # number of expert outputs

        # Launch Triton kernel: one program per (i, h)
        grid = (T, H)
        scatter_add_rows_cols_kernel[grid](
            output, expert_outputs, token_indices,
            B, H, T,
            num_warps=1,   # simple per-element kernel
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
