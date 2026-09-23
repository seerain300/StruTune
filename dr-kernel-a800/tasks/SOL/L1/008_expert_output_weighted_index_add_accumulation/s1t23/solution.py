import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_per_row_kernel(
    out_ptr,          # *bfloat16, shape [M, H]
    src_ptr,          # *bfloat16, shape [N, H]
    indices_ptr,      # *int32,    shape [N]
    M: tl.constexpr,  # total output rows
    H: tl.constexpr,  # hidden size
    N: tl.constexpr,  # number of source rows
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # One program handles one output row (pid = output row index)
    pid = tl.program_id(0)
    if pid >= M:
        return

    # Initialize accumulator for this output row
    acc = tl.zeros((BLOCK_H,), dtype=tl.bfloat16)

    # Vector of hidden offsets for the tile
    offs = tl.arange(0, BLOCK_H)

    # Scan all source rows; accumulate contributions where index == pid
    for j in range(0, N):
        idx_j = tl.load(indices_ptr + j)  # int32
        # If this source contributes to this output row
        # Note: Equality between scalar idx_j and vector pid is fine; Triton handles it.
        contrib = tl.load(src_ptr + j * H + offs, mask=offs < H, other=0.0)
        # Zero out contributions for other rows
        # Using where: where(idx_j == pid, contrib, 0)
        # Triton supports scalar comparison in where
        mask_j = (idx_j == pid)
        contrib = tl.where(mask_j, contrib, tl.zeros_like(contrib))

        # Accumulate in the current tile
        acc += contrib

    # Atomically add accumulated tile into output
    out_row_base = out_ptr + pid * H
    for start in range(0, H, BLOCK_H):
        h = start + offs
        mask = h < H
        tl.atomic_add(out_row_base + h, acc, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Clone to preserve the original buffer
        output = final_hidden_states.clone()

        # Ensure dtypes and contiguity
        assert output.dtype == torch.bfloat16, "final_hidden_states must be bfloat16"
        assert expert_outputs.dtype == torch.bfloat16, "expert_outputs must be bfloat16"
        assert token_indices.dtype in (torch.int64, torch.int32), "token_indices must be integer type"
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Make tensors contiguous for simple pointer arithmetic
        output = output.contiguous()
        src = expert_outputs.contiguous()
        indices = token_indices.contiguous()

        # Shapes
        M = output.shape[0]
        H = output.shape[1]
        N = src.shape[0]

        # Kernel launch configuration
        BLOCK_H = 256  # process H in tiles of 256; single pass for H <= 1024
        grid = (M,)    # one program per output row
        num_warps = 4
        num_stages = 2

        # Launch kernel: output, src, indices, M, H, N, BLOCK_H
        scatter_add_per_row_kernel[grid](
            output, src, indices,
            M, H, N,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return output


def run(*args):
    return ModelNew()(*args)
