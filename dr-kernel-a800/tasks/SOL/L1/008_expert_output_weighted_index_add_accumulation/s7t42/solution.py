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
        return  # safety if indices are out of bounds (shouldn't happen with provided inputs)

    # Process hidden dimension in chunks of BLOCK_SIZE
    # Masking handles cases where H is not divisible by BLOCK_SIZE.
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H

        # Compute pointers for this chunk
        # out is [N, H], contiguous, so row dst starts at dst * H
        out_row_base = dst * H
        src_row_base = pid * H

        out_ptrs = out_ptr + out_row_base + offs
        src_ptrs = src_ptr + src_row_base + offs

        # Load source values for this chunk (masked)
        src_vals = tl.load(src_ptrs, mask=mask, other=0.0)

        # Atomic add to the corresponding positions in out
        tl.atomic_add(out_ptrs, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-based scatter add: for each i in [0, M), output[token_indices[i]] += expert_outputs[i].
        Input shapes:
          - final_hidden_states: [batch_seq_len, hidden_size], dtype bfloat16
          - expert_outputs: [num_selected_tokens, hidden_size], dtype bfloat16
          - token_indices: [num_selected_tokens], dtype long (we'll cast to int32 for Triton)
        """
        # Ensure contiguity
        out = final_hidden_states.contiguous()
        src = expert_outputs.contiguous()

        # Triton prefers int32 indices
        idx = token_indices.to(torch.int32)

        # Shapes
        M = src.shape[0]
        N = out.shape[0]
        H = out.shape[1]

        # Launch Triton kernel: one program per source row
        grid = (M,)
        scatter_add_per_row_chunked_kernel[grid](
            out, src, idx, M, N, H,
            BLOCK_SIZE=256,  # chunk size along hidden dimension
            num_warps=2,     # modest parallelism per program
            num_stages=1
        )

        # out now holds the accumulated results; return it (matches reference behavior)
        return out


def run(*args):
    return ModelNew()(*args)
