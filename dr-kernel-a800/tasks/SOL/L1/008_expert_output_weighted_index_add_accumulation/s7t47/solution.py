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

    # Base pointers for this destination row
    out_row_ptr = out_ptr + dst * H
    src_row_ptr = src_ptr + pid * H

    # Process hidden dimension in chunks of BLOCK_SIZE
    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        # Load chunk from src row (masked for tail)
        val = tl.load(src_row_ptr + offs, mask=mask, other=0.0)  # bfloat16
        # Atomic add into out row (masked)
        tl.atomic_add(out_row_ptr + offs, val, mask=mask)
        start += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton atomic scatter-add:
            output[token_indices[i]] += expert_outputs[i], per hidden dimension, for all i.
        Assumes final_hidden_states and expert_outputs are bfloat16 and contiguous.
        token_indices are int64 by default; we cast to int32 for Triton.
        """
        # Ensure inputs are on CUDA
        if not final_hidden_states.is_cuda:
            raise RuntimeError("ModelNew.forward expects CUDA tensors.")
        if not expert_outputs.is_cuda:
            raise RuntimeError("ModelNew.forward expects CUDA tensors.")
        if not token_indices.is_cuda:
            raise RuntimeError("ModelNew.forward expects CUDA tensors.")

        # Ensure contiguity
        out = final_hidden_states.contiguous()
        src = expert_outputs.contiguous()
        # Indices must be int32 for Triton
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        M = src.shape[0]
        N = out.shape[0]
        H = out.shape[1]

        # Heuristic: use larger BLOCK_SIZE for larger H
        BLOCK_SIZE = 256 if H >= 256 else 128
        grid = (M,)

        # Launch kernel
        scatter_add_per_row_chunked_kernel[grid](
            out, src, token_indices,
            M, N, H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
