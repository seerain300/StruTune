import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_dim0_bf16_kernel(
    output_ptr,       # *bf16, shape [M, H]
    indices_ptr,      # *int32, shape [N]
    inputs_ptr,       # *bf16, shape [N, H]
    N,                # int32
    stride_out_row,   # int64 (elements between rows of output)
    stride_out_col,   # int64 (elements between cols of output)
    stride_in_row,    # int64 (elements between rows of inputs)
    stride_in_col,    # int64 (elements between cols of inputs)
    H: tl.constexpr,  # hidden_size (compile-time specialization)
):
    # One program per row i
    i = tl.program_id(axis=0)
    if i >= N:
        return

    # Load target row index (int32)
    idx = tl.load(indices_ptr + i)

    # Load the expert vector for row i (length H), bfloat16
    in_ptrs = inputs_ptr + i * stride_in_row + tl.arange(0, H) * stride_in_col
    vals = tl.load(in_ptrs)  # vector length H, bfloat16

    # Atomic add into output row idx across columns
    out_ptrs = output_ptr + idx * stride_out_row + tl.arange(0, H) * stride_out_col
    tl.atomic_add(out_ptrs, vals)


@triton.jit
def scatter_add_dim0_fp32_kernel(
    output_ptr,       # *fp32, shape [M, H]
    indices_ptr,      # *int32, shape [N]
    inputs_ptr,       # *fp32, shape [N, H]
    N,                # int32
    stride_out_row,   # int64 (elements between rows of output)
    stride_out_col,   # int64 (elements between cols of output)
    stride_in_row,    # int64 (elements between rows of inputs)
    stride_in_col,    # int64 (elements between cols of inputs)
    H: tl.constexpr,  # hidden_size (compile-time specialization)
):
    # One program per row i
    i = tl.program_id(axis=0)
    if i >= N:
        return

    # Load target row index (int32)
    idx = tl.load(indices_ptr + i)

    # Load the expert vector for row i (length H), fp32
    in_ptrs = inputs_ptr + i * stride_in_row + tl.arange(0, H) * stride_in_col
    vals = tl.load(in_ptrs)  # vector length H, fp32

    # Atomic add into output row idx across columns
    out_ptrs = output_ptr + idx * stride_out_row + tl.arange(0, H) * stride_out_col
    tl.atomic_add(out_ptrs, vals)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Preserve original semantics: clone the accumulator (random init)
        output = final_hidden_states.clone()

        # Ensure CUDA execution
        if (not final_hidden_states.is_cuda) or (not expert_outputs.is_cuda) or (not token_indices.is_cuda):
            raise RuntimeError("All tensors must be on CUDA for Triton execution.")

        M, H = output.shape
        N = expert_outputs.shape[0]
        assert expert_outputs.shape == (N, H), "expert_outputs must have shape (num_selected_tokens, hidden_size)"
        assert token_indices.shape[0] == N, "token_indices length must match expert_outputs rows"

        # Ensure inputs are contiguous
        expert_outputs = expert_outputs.contiguous()
        # Triton prefers int32 indices
        token_indices = token_indices.to(torch.int32).contiguous()

        # Strides in elements
        stride_out_row = output.stride(0)
        stride_out_col = output.stride(1)
        stride_in_row = expert_outputs.stride(0)
        stride_in_col = expert_outputs.stride(1)

        # Try bf16 atomic path first
        try:
            grid = (N,)
            scatter_add_dim0_bf16_kernel[grid](
                output, token_indices, expert_outputs,
                N,
                stride_out_row, stride_out_col,
                stride_in_row, stride_in_col,
                H=H,
                num_warps=4,
                num_stages=1,
            )
            return output
        except Exception:
            # Fallback: compute in fp32 via Triton, then cast back to bfloat16
            output_fp32 = output.to(torch.float32).contiguous()
            expert_outputs_fp32 = expert_outputs.to(torch.float32).contiguous()

            stride_out_row_fp32 = output_fp32.stride(0)
            stride_out_col_fp32 = output_fp32.stride(1)
            stride_in_row_fp32 = expert_outputs_fp32.stride(0)
            stride_in_col_fp32 = expert_outputs_fp32.stride(1)

            grid = (N,)
            scatter_add_dim0_fp32_kernel[grid](
                output_fp32, token_indices, expert_outputs_fp32,
                N,
                stride_out_row_fp32, stride_out_col_fp32,
                stride_in_row_fp32, stride_in_col_fp32,
                H=H,
                num_warps=4,
                num_stages=1,
            )
            # Cast back to bfloat16 to match original dtype
            return output_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
