import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *bfloat16, shape [M, H]
    src_ptr,          # *bfloat16, shape [N, H]
    idx_ptr,          # *int32,    shape [N]
    M: tl.constexpr,  # batch_seq_len
    H: tl.constexpr,  # hidden_size
    N: tl.constexpr,  # num_selected_tokens
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # One program per source row
    row = tl.program_id(axis=0)
    if row >= N:
        return

    # Destination token index for this row
    dest = tl.load(idx_ptr + row)
    if dest < 0 or dest >= M:
        return  # defensive guard

    # Iterate over hidden dimension in tiles
    for j in range(0, H, BLOCK_H):
        cols = j + tl.arange(0, BLOCK_H)
        mask = cols < H

        # Compute linear offsets in row-major layout
        out_offs = dest * H + cols
        src_offs = row * H + cols

        # Load source vector (masked for cols beyond H)
        val = tl.load(src_ptr + src_offs, mask=mask, other=0.0)

        # Atomic add into output (masked)
        tl.atomic_add(out_ptr + out_offs, val, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Shapes
        M, H = final_hidden_states.shape
        N = expert_outputs.shape[0]
        assert token_indices.shape[0] == N, "token_indices must have shape [num_selected_tokens]"

        # Clone to match original behavior
        output = final_hidden_states.clone()

        # Ensure contiguity and dtypes
        src = expert_outputs.contiguous()
        idx32 = token_indices.to(torch.int32).contiguous()

        # Launch Triton kernel: one program per source row
        grid = (N,)

        # Use a larger tile to reduce loop iterations; works well for H up to 1024
        BLOCK_H = 256

        scatter_add_rows_kernel[grid](
            output,
            src,
            idx32,
            M=M,
            H=H,
            N=N,
            BLOCK_H=BLOCK_H,
            num_warps=8,     # more warps for better throughput on larger tiles
            num_stages=2,    # allow some pipelining of memory ops
        )

        return output


def run(*args):
    return ModelNew()(*args)
