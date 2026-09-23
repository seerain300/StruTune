import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 512}, num_warps=4, num_stages=2),
    ],
    key=['H'],
)
@triton.jit
def _scatter_add_rows_kernel(
    out_ptr,        # *bf16, (M, H)
    in_ptr,         # *bf16, (N, H)
    idx_ptr,        # *int32, (N,)
    M,              # int: number of rows in output (batch_seq_len)
    N,              # int: number of rows in input (num_selected_tokens)
    H: tl.constexpr,  # hidden size (compile-time for autotune key)
    BLOCK: tl.constexpr,
):
    # One program per input row
    row_id = tl.program_id(0)
    # Guard: if row_id >= N, return (safety, though grid should match N)
    if row_id >= N:
        return

    # Load the token index for this row
    dest_row = tl.load(idx_ptr + row_id)

    # Vector of column offsets for a chunk
    cols = tl.arange(0, BLOCK)

    # Iterate across the hidden dimension in chunks
    # Note: H is constexpr here, so loop is unrolled or compiled efficiently
    for start in range(0, H, BLOCK):
        offs = start + cols
        mask = offs < H

        # Load the chunk of the expert row
        in_offsets = row_id * H + offs
        vals = tl.load(in_ptr + in_offsets, mask=mask, other=0.0)

        # Compute output offsets for this chunk and atomically add
        out_offsets = dest_row * H + offs
        # Atomic add into output for this chunk
        tl.atomic_add(out_ptr + out_offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized version of:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        # Ensure contiguous tensors
        out = final_hidden_states.contiguous()
        in_t = expert_outputs.contiguous()
        # Triton prefers int32 indices for pointer arithmetic
        idx = token_indices.to(torch.int32).contiguous()

        M = out.shape[0]           # batch_size * seq_len
        N = in_t.shape[0]          # num_selected_tokens
        H = out.shape[1]           # hidden_size

        # Launch one program per row
        grid = (N,)

        # Run the Triton kernel
        _scatter_add_rows_kernel[grid](
            out, in_t, idx,
            M, N, H=H,  # pass H as constexpr for autotune key
        )

        return out


def run(*args):
    return ModelNew()(*args)
