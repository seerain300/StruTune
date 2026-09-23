import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_add_rows_kernel(
    out_ptr,            # *bf16, (M, H)
    in_ptr,             # *bf16, (N, H)
    idx_ptr,            # *int32, (N,)
    M,                  # int: number of rows in output (batch_size * seq_len)
    N,                  # int: number of rows in in_ptr/idx_ptr (num_selected_tokens)
    H,                  # int: hidden size
    stride_out_row: tl.constexpr,  # row stride of out (elements)
    stride_out_col: tl.constexpr,  # col stride of out (elements, usually 1)
    stride_in_row: tl.constexpr,   # row stride of in (elements, usually 1)
    stride_in_col: tl.constexpr,   # col stride of in (elements, usually 1)
    BLOCK: tl.constexpr,           # chunk size across hidden dimension
):
    # One Triton program per row i in [0, N)
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Load destination row index for this row
    idx = tl.load(idx_ptr + pid)
    if idx < 0 or idx >= M:
        return

    # Base pointers for the source row and destination row
    in_row_base = in_ptr + pid * stride_in_row
    out_row_base = out_ptr + idx * stride_out_row

    # Vectorized chunked accumulation across hidden dimension H
    col_offsets = tl.arange(0, BLOCK)
    for start in range(0, H, BLOCK):
        cols = start + col_offsets
        mask = cols < H

        # Load the values from expert_outputs[pid, cols]
        in_ptrs = in_row_base + cols * stride_in_col
        vals = tl.load(in_ptrs, mask=mask, other=0.0)

        # Atomic add into output[idx, cols]
        out_ptrs = out_row_base + cols * stride_out_col
        tl.atomic_add(out_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add equivalent to:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        This performs the accumulation using Triton kernels with atomic adds.
        """
        # Clone to match PyTorch semantics
        output = final_hidden_states.clone()

        # Ensure tensors are CUDA and contiguous
        assert output.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton"
        assert output.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Tensors must be bfloat16 (bf16)"
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()

        # token_indices: Triton prefers int32 for pointer arithmetic
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        M = output.shape[0]
        N = expert_outputs.shape[0]
        H = output.shape[1]

        # Choose BLOCK based on H for better performance
        BLOCK = 256 if H >= 256 else 128
        num_warps = 8 if BLOCK == 256 else 4

        # Launch one program per row
        grid = (N,)

        _scatter_add_rows_kernel[grid](
            output, expert_outputs, token_indices,
            M, N, H,
            output.stride(0), output.stride(1),
            expert_outputs.stride(0), expert_outputs.stride(1),
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
