import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_dim0_row_kernel(
    output_ptr,       # *bf16, shape [M, H]
    indices_ptr,      # *int32, shape [N]
    inputs_ptr,       # *bf16, shape [N, H]
    N,                # int32
    stride_out_row,   # int64 (elements between rows of output)
    stride_out_col,   # int64 (elements between cols of output, typically 1)
    stride_in_row,    # int64 (elements between rows of inputs, typically H)
    stride_in_col,    # int64 (elements between cols of inputs, typically 1)
    H: tl.constexpr,  # hidden_size (compile-time for vectorization)
    BLOCK: tl.constexpr,  # number of columns processed per program (>= H)
):
    # One program per row i
    i = tl.program_id(axis=0)
    if i >= N:
        return

    # Load target row index (int32)
    idx = tl.load(indices_ptr + i)  # scalar int32

    # Column offsets and mask
    cols = tl.arange(0, BLOCK)
    mask = cols < H  # ensure we only process valid hidden dims

    # Load expert vector for row i across H columns
    in_ptrs = inputs_ptr + i * stride_in_row + cols * stride_in_col
    vals = tl.load(in_ptrs, mask=mask, other=0.0)  # shape (BLOCK,), bf16

    # Compute output pointers for row idx and atomic add vals across columns
    out_ptrs = output_ptr + idx * stride_out_row + cols * stride_out_col
    tl.atomic_add(out_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure we run the Triton kernel on GPU. No CPU fallback on CUDA tensors.
        # If tensors are not on CUDA, raise to avoid silent incorrectness.
        if (not final_hidden_states.is_cuda) or (not expert_outputs.is_cuda) or (not token_indices.is_cuda):
            raise RuntimeError("ModelNew.forward requires CUDA tensors for Triton execution.")

        # Clone the accumulator to preserve original random initialization
        output = final_hidden_states.clone().contiguous()

        # Shapes and assertions
        M, H = output.shape
        N = expert_outputs.shape[0]
        assert expert_outputs.shape == (N, H), "expert_outputs must have shape (num_selected_tokens, hidden_size)"
        assert token_indices.shape[0] == N, "token_indices length must match expert_outputs rows"

        # Ensure inputs are contiguous
        expert_outputs = expert_outputs.contiguous()
        # Cast indices to int32 for pointer arithmetic
        token_indices = token_indices.to(torch.int32).contiguous()

        # Strides in elements
        stride_out_row = output.stride(0)
        stride_out_col = output.stride(1)
        stride_in_row = expert_outputs.stride(0)  # typically H
        stride_in_col = expert_outputs.stride(1)  # typically 1

        # Launch Triton kernel: one program per row i
        grid = (N,)

        # Choose BLOCK as the next power of two >= H, capped to 1024 for safety.
        # Since H is usually 128 or 256 in the provided configs, BLOCK=256 is sufficient.
        BLOCK = 256 if H <= 256 else 1024

        scatter_add_dim0_row_kernel[grid](
            output, token_indices, expert_outputs,
            N,
            stride_out_row, stride_out_col,
            stride_in_row, stride_in_col,
            H=H, BLOCK=BLOCK,
            num_warps=4,   # tuneable: 4 or 8 are typical
            num_stages=2   # tuneable
        )

        return output


def run(*args):
    return ModelNew()(*args)
