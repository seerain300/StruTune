import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_dim0_rows_kernel(
    output_ptr,       # *bf16, shape [M, H]
    indices_ptr,      # *int32, shape [N]
    inputs_ptr,       # *bf16, shape [N, H] (expert_outputs)
    N,                # int32
    stride_out_row,   # int64 (elements between rows of output)
    stride_out_col,   # int64 (elements between cols of output, typically 1)
    stride_in_row,    # int64 (elements between rows of inputs, typically H)
    stride_in_col,    # int64 (elements between cols of inputs, typically 1)
    H: tl.constexpr,  # hidden size, compile-time for vectorization
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)  # process BLOCK rows of expert_outputs
    mask = offs < N

    # Load target row indices (int32): one index per row i
    idx = tl.load(indices_ptr + offs, mask=mask, other=0)

    # Load the entire vector for each row i from inputs_ptr
    # inputs_ptr points to expert_outputs; we add row i across all H columns
    in_ptrs = inputs_ptr + offs[:, None] * stride_in_row + tl.arange(0, H)[None, :] * stride_in_col
    vals = tl.load(in_ptrs, mask=mask[:, None], other=0.0)

    # Compute output pointers for each row i and each column j: output[idx[i], j]
    out_ptrs = output_ptr + idx[:, None] * stride_out_row + tl.arange(0, H)[None, :] * stride_out_col

    # Atomic add: add vals[i, :] into output[row=idx[i], :]
    tl.atomic_add(out_ptrs, vals, mask=mask[:, None])


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Triton requires CUDA tensors
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."

        # Preserve original semantics: clone the accumulator (random init)
        output = final_hidden_states.clone().contiguous()

        # Shapes and assertions
        M, H = output.shape
        N = expert_outputs.shape[0]
        assert expert_outputs.shape == (N, H), "expert_outputs must have shape (num_selected_tokens, hidden_size)"
        assert token_indices.shape[0] == N, "token_indices length must match expert_outputs rows"

        # Ensure inputs are contiguous
        expert_outputs = expert_outputs.contiguous()
        # Cast indices to int32 to reduce bandwidth (M is within 32-bit range for typical sizes)
        token_indices = token_indices.to(torch.int32).contiguous()

        # Strides in elements
        stride_out_row = output.stride(0)
        stride_out_col = output.stride(1)
        stride_in_row = expert_outputs.stride(0)  # typically H
        stride_in_col = expert_outputs.stride(1)  # typically 1

        # Launch Triton kernel
        BLOCK = 1024  # rows per program; tune if needed
        grid = (triton.cdiv(N, BLOCK),)

        scatter_add_dim0_rows_kernel[grid](
            output, token_indices, expert_outputs,  # inputs_ptr must be expert_outputs (the source to add)
            N,
            stride_out_row, stride_out_col,
            stride_in_row, stride_in_col,
            H=H,           # hidden_size as compile-time constexpr for vectorization
            BLOCK=BLOCK,
            num_warps=4,   # reasonable default; can try 8 for larger H/N
            num_stages=2,  # small kernel; 2 stages fine
        )

        return output


def run(*args):
    return ModelNew()(*args)
