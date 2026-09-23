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
    BLOCK_SIZE: tl.constexpr,  # chunk size along H (compile-time)
):
    # One program per source row
    pid = tl.program_id(axis=0)
    if pid >= M:
        return

    # Destination row index in out
    dst = tl.load(indices_ptr + pid)  # int32
    if dst < 0 or dst >= N:
        return  # safety (indices should be valid)

    # Process hidden dimension in chunks of BLOCK_SIZE
    # Masking handles cases where H is not divisible by BLOCK_SIZE.
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H

        # Load source row chunk (bf16)
        src_row_ptr = src_ptr + pid * H
        src_vals = tl.load(src_row_ptr + offs, mask=mask, other=0.0)

        # Compute output row pointers for this chunk
        out_row_ptr = out_ptr + dst * H
        out_vals = tl.load(out_row_ptr + offs, mask=mask, other=0.0)

        # Atomic add contribution
        tl.atomic_add(out_row_ptr + offs, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        """
        Triton-optimized atomic scatter-add:
            output[token_indices[i]] += expert_outputs[i]
        Args:
            final_hidden_states: Tensor of shape (batch_seq_len, hidden_size), dtype bfloat16
            expert_outputs: Tensor of shape (num_selected_tokens, hidden_size), dtype bfloat16
            token_indices: Tensor of shape (num_selected_tokens,), dtype long
        Returns:
            updated final_hidden_states
        """
        # Ensure contiguity and correct dtypes; no torch ops for the actual accumulation
        out = final_hidden_states.contiguous()
        src = expert_outputs.contiguous()
        indices = token_indices.contiguous()
        if indices.dtype != torch.int32:
            indices = indices.to(torch.int32)

        M = src.shape[0]
        N = out.shape[0]
        H = out.shape[1]

        # Launch one program per source row
        grid = (M,)

        # Micro-tuning: larger block/warps for larger H
        BLOCK_SIZE = 256 if H >= 256 else 128
        num_warps = 2 if H >= 256 else 1

        scatter_add_per_row_chunked_kernel[grid](
            out, src, indices, M, N, H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=1,
        )
        return out


def run(*args):
    return ModelNew()(*args)
