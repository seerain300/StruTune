import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_per_row_chunked_kernel(
    out_ptr,        # *bf16, pointer to out tensor [N, H], contiguous
    src_ptr,        # *bf16, pointer to src tensor [M, H], contiguous
    indices_ptr,    # *int32, pointer to indices tensor [M], contiguous
    M,              # int32, number of source rows
    N,              # int32, number of rows in out (batch_seq_len)
    H,              # int32, hidden size
    BLOCK_SIZE: tl.constexpr,  # chunk size along H (compile-time constant)
):
    # One program per source row
    pid = tl.program_id(axis=0)
    if pid >= M:
        return

    # Destination row in out
    dst = tl.load(indices_ptr + pid)  # int32

    # Process hidden dimension in chunks of BLOCK_SIZE
    # Masking handles cases where H is not divisible by BLOCK_SIZE.
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        # Compute pointers for this row
        out_row_ptr = out_ptr + dst * H
        src_row_ptr = src_ptr + pid * H
        # Load source values for this chunk
        vals = tl.load(src_row_ptr + offs, mask=mask, other=0.0)
        # Atomic add into destination
        tl.atomic_add(out_row_ptr + offs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton version of: output[token_indices[i]] += expert_outputs[i]
        Returns the updated tensor (does not modify the input in-place).
        """
        # Ensure tensors are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA for Triton."

        # Allocate output tensor; we do not modify the input in-place.
        out = torch.empty_like(final_hidden_states)

        M = expert_outputs.shape[0]
        N = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]

        # Triton prefers int32 indices
        token_indices_i32 = token_indices if token_indices.dtype == torch.int32 else token_indices.to(torch.int32)

        # Choose a moderate chunk size and warps for reliability and performance
        BLOCK_SIZE = 128
        grid = (M,)

        scatter_add_per_row_chunked_kernel[grid](
            out, expert_outputs.contiguous(), token_indices_i32,
            M, N, H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=2,
            num_stages=1,
        )
        return out


def run(*args):
    return ModelNew()(*args)
