import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,    shape [N]
    M: tl.constexpr,  # total rows in out_ptr/src_ptr (kept for clarity)
    H: tl.constexpr,  # hidden size
    BLOCK_H: tl.constexpr,  # tile size across H
):
    # Each program handles one source row i
    pid = tl.program_id(0)  # i in [0, N)

    # Loop over hidden dimension in tiles of size BLOCK_H
    for h_start in range(0, H, BLOCK_H):
        offs = h_start + tl.arange(0, BLOCK_H)
        mask = offs < H

        # Destination index for this row (int32)
        index = tl.load(indices_ptr + pid, mask=True, other=0)  # scalar int32

        # Compute output and source pointers for this row and tile
        out_row_ptr = out_ptr + pid * H
        out_dst_ptr = out_ptr + index * H + offs

        src_row_ptr = src_ptr + pid * H
        vals = tl.load(src_row_ptr + offs, mask=mask, other=0.0)  # bfloat16

        # Atomic add contribution into output
        tl.atomic_add(out_dst_ptr, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        final_hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        # Ensure contiguity and dtype for Triton
        out = final_hidden_states.contiguous()
        src = expert_outputs.contiguous()
        indices = token_indices.to(torch.int32).contiguous()

        M = out.shape[0]
        H = out.shape[1]
        N = src.shape[0]

        # One program per source row
        grid = (N,)

        # Heuristic configuration based on H to balance occupancy and loop iterations
        if H <= 256:
            BLOCK_H = 256
            num_warps = 4
            num_stages = 2
        elif H <= 512:
            BLOCK_H = 512
            num_warps = 4
            num_stages = 2
        else:
            BLOCK_H = 256
            num_warps = 4
            num_stages = 2

        scatter_add_rows_kernel[grid](
            out, src, indices,
            M, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out


def run(*args):
    return ModelNew()(*args)
