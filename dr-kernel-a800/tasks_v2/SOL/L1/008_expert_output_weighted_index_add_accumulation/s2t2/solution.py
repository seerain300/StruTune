import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_dim0_block_kernel(
    output_ptr,       # *bf16, shape [M, H]
    indices_ptr,      # *int32, shape [N]
    inputs_ptr,       # *bf16, shape [N, H] (expert_outputs)
    N,                # int32
    H: tl.constexpr,  # hidden size (compile-time for vectorization)
    stride_out_row,   # int64 (elements between rows of output)
    stride_out_col,   # int64 (elements between cols of output)
    stride_in_row,    # int64 (elements between rows of inputs)
    stride_in_col,    # int64 (elements between cols of inputs)
    BLOCK: tl.constexpr,  # number of rows processed per program
):
    # 1D grid: each program handles BLOCK rows
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)  # row indices to process
    mask = offs < N

    # Load target row indices for this block (int32)
    idx = tl.load(indices_ptr + offs, mask=mask, other=0)

    # Load the expert vectors for these rows (length H), bfloat16
    in_ptrs = inputs_ptr + offs[:, None] * stride_in_row + tl.arange(0, H)[None, :] * stride_in_col
    vals = tl.load(in_ptrs, mask=mask[:, None], other=0.0)  # shape (BLOCK, H), bfloat16

    # Compute output pointers for these rows and columns
    out_ptrs = output_ptr + idx[:, None] * stride_out_row + tl.arange(0, H)[None, :] * stride_out_col

    # Atomic add: add vals into output[row=idx[i], col=tl.arange(H)]
    tl.atomic_add(out_ptrs, vals, mask=mask[:, None])


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Fallback to PyTorch if not on CUDA to ensure correctness
        if (not final_hidden_states.is_cuda) or (not expert_outputs.is_cuda) or (not token_indices.is_cuda):
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return output

        # Preserve original semantics: clone the accumulator (random init)
        output = final_hidden_states.clone().contiguous()

        # Shapes and assertions
        M, H = output.shape
        N = expert_outputs.shape[0]
        assert expert_outputs.shape == (N, H), "expert_outputs must have shape (num_selected_tokens, hidden_size)"
        assert token_indices.shape[0] == N, "token_indices length must match expert_outputs rows"

        # Defensive: if N == 0, nothing to do
        if N == 0:
            return output

        # Ensure inputs are contiguous
        expert_outputs = expert_outputs.contiguous()
        # Cast indices to int32 to reduce bandwidth and simplify pointer arithmetic
        token_indices = token_indices.to(torch.int32).contiguous()

        # Strides in elements
        stride_out_row = output.stride(0)
        stride_out_col = output.stride(1)
        stride_in_row = expert_outputs.stride(0)  # typically H
        stride_in_col = expert_outputs.stride(1)  # typically 1

        # Launch Triton kernel: one program per BLOCK rows
        BLOCK = 256  # rows per program; can tune (128/256/512). 256 is a robust default.
        grid = (triton.cdiv(N, BLOCK),)

        scatter_add_dim0_block_kernel[grid](
            output, token_indices, expert_outputs,
            N,
            H=H,
            stride_out_row=stride_out_row, stride_out_col=stride_out_col,
            stride_in_row=stride_in_row, stride_in_col=stride_in_col,
            BLOCK=BLOCK,
            num_warps=4,   # reasonable default; tune for larger N/GPUs
            num_stages=2,  # small kernel; 2 stages fine
        )

        return output


def run(*args):
    return ModelNew()(*args)
