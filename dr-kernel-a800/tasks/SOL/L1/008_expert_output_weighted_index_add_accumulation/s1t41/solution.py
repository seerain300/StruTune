import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,        # *const bfloat16, shape [M, H]
    src_ptr,        # *const bfloat16, shape [N, H]
    indices_ptr,    # *const int32,    shape [N]
    M: tl.constexpr,    # number of rows in output (M = batch_size * seq_len)
    H: tl.constexpr,    # hidden size (columns)
    N: tl.constexpr,    # number of source rows (N = M * num_experts_per_tok)
    BLOCK_H: tl.constexpr,  # tile size along hidden dim
):
    # One program per source row
    i = tl.program_id(0)
    if i >= N:
        return

    # Load the token index for this row (int32)
    idx = tl.load(indices_ptr + i)

    # Iterate over hidden dimension in tiles of size BLOCK_H
    for h_off in range(0, H, BLOCK_H):
        offs = h_off + tl.arange(0, BLOCK_H)
        mask = offs < H

        # Load src[i, offs]
        src_row_ptr = src_ptr + i * H
        vals = tl.load(src_row_ptr + offs, mask=mask, other=0.0)  # bfloat16

        # Atomic add into out[idx, offs]
        out_row_ptr = out_ptr + idx * H
        tl.atomic_add(out_row_ptr + offs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-based implementation of:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        All accumulation is done by Triton with proper atomics.
        """
        # Start with a clone to match reference semantics
        out = final_hidden_states.clone()

        # Triton prefers int32 for indices
        indices_i32 = token_indices.to(torch.int32)
        # Ensure source tensor is contiguous
        src = expert_outputs.contiguous()

        # Shapes
        M, H = out.shape
        N = src.shape[0]

        # Launch Triton kernel: one program per source row
        grid = (N,)

        # Choose robust tile size and launch configuration
        BLOCK_H = 256
        scatter_add_rows_kernel[grid](
            out, src, indices_i32,
            M=M, H=H, N=N,
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
