import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_blocks_kernel(
    output_ptr,     # *fp16/bf16/fp32
    expert_ptr,     # *fp16/bf16/fp32
    indices_ptr,    # *int64
    N, H,           # int32
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 2D launch: programs over (rows of expert_outputs, tiles of H)
    row_block = tl.program_id(0)
    h_block = tl.program_id(1)

    n_offsets = row_block * BLOCK_N + tl.arange(0, BLOCK_N)            # [BLOCK_N]
    h_offsets = h_block * BLOCK_H + tl.arange(0, BLOCK_H)              # [BLOCK_H]

    mask_n = n_offsets < N
    mask_h = h_offsets < H

    # Load token indices for these N rows
    idx = tl.load(indices_ptr + n_offsets, mask=mask_n, other=0)       # [BLOCK_N], int64

    # Compute destination linear offsets for 2D tile (BLOCK_N x BLOCK_H)
    dest_offsets = idx[:, None] * H + h_offsets[None, :]               # [BLOCK_N, BLOCK_H]
    valid = mask_n[:, None] & mask_h[None, :]

    # Load expert_outputs for the same rows and H block
    vals = tl.load(
        expert_ptr + n_offsets[:, None] * H + h_offsets[None, :],
        mask=valid,
        other=0.0,
    )  # [BLOCK_N, BLOCK_H]

    # Atomic add into output at destination rows
    tl.atomic_add(output_ptr + dest_offsets, vals, mask=valid)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        """
        Triton-optimized forward: scatter-add expert_outputs into output according to token_indices.
        output[i] += expert_outputs[j] where token_indices[j] == i, for all j.
        """
        # Ensure tensors are on the same device and dtype (final_hidden_states dtype)
        # Triton kernels commonly handle bf16/fp32; we keep dtype as is.
        # Clone to initialize output (PyTorch does not have no-op random init, so clone is fine).
        output = final_hidden_states.clone()

        # Shapes
        M, H = output.shape
        N = expert_outputs.shape[0]

        # Ensure inputs are contiguous
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Choose tiling parameters based on H
        if H >= 1024:
            BLOCK_H = 128
            num_warps = 4
            num_stages = 3
        elif H >= 512:
            BLOCK_H = 128
            num_warps = 4
            num_stages = 2
        elif H >= 256:
            BLOCK_H = 64
            num_warps = 4
            num_stages = 2
        else:
            BLOCK_H = 32
            num_warps = 2
            num_stages = 2

        BLOCK_N = 64 if N >= 64 else 32

        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(H, BLOCK_H))

        scatter_add_blocks_kernel[grid](
            output, expert_outputs, token_indices,
            N, H,
            BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=num_stages,
        )

        return output


def run(*args):
    return ModelNew()(*args)
