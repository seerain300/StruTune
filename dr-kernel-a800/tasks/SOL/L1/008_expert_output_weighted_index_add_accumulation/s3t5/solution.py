import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_atomic_rows_kernel(
    out_ptr,        # *fp32, shape [batch_seq_len, H]
    A_ptr,          # *fp32, shape [N, H]
    idx_ptr,        # *int32, shape [N]
    N,              # int32
    H: tl.constexpr # hidden size as compile-time constant for vectorization
):
    pid = tl.program_id(0)
    # Load destination row index for this token
    row_idx = tl.load(idx_ptr + pid)
    # Cast to 64-bit for pointer arithmetic
    row_idx64 = row_idx.to(tl.int64)

    # Column offsets
    cols = tl.arange(0, H)
    cols64 = cols.to(tl.int64)

    # Base pointers for output row and A row
    out_row_ptr = out_ptr + row_idx64 * H
    A_row_ptr = A_ptr + pid * H  # A is [N, H] contiguous

    # Atomic add the A row into the output row
    tl.atomic_add(out_row_ptr + cols64, tl.load(A_row_ptr + cols64))


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure tensors are on CUDA
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."

        # Shapes
        batch_seq_len, H = final_hidden_states.shape
        N = expert_outputs.shape[0]
        assert token_indices.numel() == N, "token_indices length must equal number of expert outputs."

        # Make inputs contiguous
        final_hidden_states = final_hidden_states.contiguous()
        token_indices = token_indices.contiguous()

        # Allocate fp32 output buffer and initialize with final_hidden_states (fp32) for identical starting state
        out_fp32 = final_hidden_states.to(torch.float32).clone()

        # Convert expert_outputs to fp32 for computation
        A_fp32 = expert_outputs.to(torch.float32).contiguous()

        # Triton expects int32 indices
        idx_i32 = token_indices.to(torch.int32).contiguous()

        # Launch one program per token
        grid = (N,)
        scatter_add_atomic_rows_kernel[grid](
            out_fp32, A_fp32, idx_i32,
            N,
            H=H,  # compile-time hidden size for vectorization
            num_warps=1,
        )

        # Cast back to bfloat16 to match original output dtype
        output = out_fp32.to(torch.bfloat16)
        return output


def run(*args):
    return ModelNew()(*args)
