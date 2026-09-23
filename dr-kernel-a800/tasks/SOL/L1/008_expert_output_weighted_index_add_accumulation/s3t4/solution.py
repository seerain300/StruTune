import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_chunked_kernel(
    out_ptr,   # *fp32, shape [M, H] (buffer to accumulate in fp32)
    A_ptr,     # *fp32, shape [N, H] (expert_outputs in fp32)
    idx_ptr,   # *int32, shape [N] (token_indices)
    N,         # int32, number of tokens
    H,         # int32, hidden size
    BLOCK_SIZE: tl.constexpr,  # chunk size for columns
):
    # One program per token i
    i = tl.program_id(0)

    # Load the target row index for this token
    idx_i = tl.load(idx_ptr + i)

    # Iterate over hidden dimension in chunks
    for h in range(0, H):
        offs = h + tl.arange(0, BLOCK_SIZE)
        mask = offs < H

        # Compute base pointers for the row
        out_row_ptr = out_ptr + idx_i * H
        A_row_ptr = A_ptr + i * H

        # Load current out chunk and expert chunk, add, and store back
        out_vals = tl.load(out_row_ptr + offs, mask=mask, other=0.0)
        v = tl.load(A_row_ptr + offs, mask=mask, other=0.0)
        out_vals += v
        tl.store(out_row_ptr + offs, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        """
        Triton-based scatter-add that mimics:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We accumulate in float32 for robustness. We do NOT modify the input
        'final_hidden_states' tensor; instead, we return the result (cast to bfloat16).
        """
        # Ensure tensors are CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA device for Triton kernel."
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        M, H = final_hidden_states.shape  # M = batch_size * seq_len
        N = expert_outputs.shape[0]

        # Prepare fp32 output buffer (clone original for accumulation)
        out_fp32 = final_hidden_states.to(torch.float32).contiguous()

        # Convert inputs to fp32 and int32 for kernel
        A_fp32 = expert_outputs.to(torch.float32).contiguous()
        idx_i32 = token_indices.to(torch.int32).contiguous()

        # Launch one program per token
        grid = (N,)
        scatter_add_rows_chunked_kernel[grid](
            out_fp32, A_fp32, idx_i32,
            N, H,
            BLOCK_SIZE=128,  # safe chunk size; loop handles any H
            num_warps=1,
        )

        # Cast back to bfloat16 to match expected dtype
        output = out_fp32.to(torch.bfloat16)
        return output


def run(*args):
    return ModelNew()(*args)
