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

    # Destination row index in out
    dst = tl.load(indices_ptr + pid)  # int32
    if dst < 0 or dst >= N:
        return  # safety (shouldn't happen with provided inputs)

    # Base pointers for this row
    base_out = out_ptr + dst * H
    base_src = src_ptr + pid * H

    # Process hidden dimension in chunks of BLOCK_SIZE
    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        # Load a chunk from src row
        val = tl.load(base_src + offs, mask=mask, other=0.0)
        # Atomic add into the destination row
        tl.atomic_add(base_out + offs, val, mask=mask)
        start += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter add:
        final_hidden_states: [batch_seq_len, hidden_size], bfloat16
        expert_outputs:       [num_selected_tokens, hidden_size], bfloat16
        token_indices:        [num_selected_tokens], int64/long
        """
        # Ensure CUDA tensors for Triton
        assert final_hidden_states.is_cuda, "final_hidden_states must be on CUDA for Triton kernel"
        assert expert_outputs.is_cuda, "expert_outputs must be on CUDA for Triton kernel"
        assert token_indices.is_cuda, "token_indices must be on CUDA for Triton kernel"

        # Ensure contiguity and dtype
        out = final_hidden_states.contiguous()
        src = expert_outputs.contiguous()
        # Triton prefers int32 indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        M = src.shape[0]
        N = out.shape[0]
        H = out.shape[1]

        # Use a moderate BLOCK_SIZE and num_warps that previously worked well
        BLOCK_SIZE = 128
        num_warps = 2

        # Launch one program per source row
        grid = (M,)
        scatter_add_per_row_chunked_kernel[grid](out, src, token_indices, M, N, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        return out


def run(*args):
    return ModelNew()(*args)
