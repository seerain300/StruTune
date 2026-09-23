import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_tiles_kernel(
    output_ptr,          # *ptr to output[M, H], dtype: bfloat16
    expert_outputs_ptr,  # *ptr to expert_outputs[N, H], dtype: bfloat16
    token_indices_ptr,   # *ptr to token_indices[N], dtype: int64
    N: tl.constexpr,     # number of source rows
    H: tl.constexpr,     # number of hidden features
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 2D grid: over tiles of rows (N) and hidden features (H)
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Offsets for this tile
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]

    # Bounds masks
    mask_n = n_offsets < N
    mask_h = h_offsets < H

    # Load target indices for each source row in this tile
    # token_indices[n] in [0, M), M = N from get_inputs here? No, M = batch_size * seq_len.
    # We need the relationship from host; kernel only uses N and H, so we'll load indices and rely on host shapes.
    idxs = tl.load(token_indices_ptr + n_offsets, mask=mask_n, other=0)  # [BLOCK_N], int64

    # Accumulator per (row, hidden-feature) within this tile
    acc = tl.zeros((BLOCK_N, BLOCK_H), dtype=tl.bfloat16)

    # Iterate over hidden features in the block
    for h_iter in range(BLOCK_H):
        # Check if this hidden feature index is within valid H
        if h_offsets[h_iter] < H:
            # Load vals_n for this hidden feature across BLOCK_N rows
            vals_n = tl.load(
                expert_outputs_ptr + n_offsets * H + h_offsets[h_iter],
                mask=mask_n,
                other=0.0,
            )  # [BLOCK_N], bfloat16
            # Accumulate into acc
            acc[:, h_iter] = vals_n

    # Compute destination addresses: idxs * H + h_offsets
    idxs_i32 = idxs.to(tl.int32)
    h_offsets_i32 = h_offsets.to(tl.int32)
    dest = idxs_i32[:, None] * H + h_offsets_i32[None, :]  # [BLOCK_N, BLOCK_H]

    # Final mask for valid rows and valid hidden offsets
    mask = mask_n[:, None] & mask_h[None, :]

    # Atomically add accumulated contributions to output
    tl.atomic_add(output_ptr + dest, acc, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add along dim=0:
        output[token_indices[i]] += expert_outputs[i] for all i.
        Returns updated output.
        """
        # Ensure device is CUDA and dtypes are bfloat16
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Inputs must be on CUDA device for Triton."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Dtypes must be bfloat16."
        assert token_indices.dtype in (torch.int64, torch.int32), "token_indices must be int64 or int32."

        # Clone the initial buffer; we won't modify the input in-place
        output = final_hidden_states.clone()

        N = expert_outputs.shape[0]
        H = expert_outputs.shape[1]

        # Heuristic tiling parameters: increase parallelism and keep register usage reasonable
        # Use larger tiles for larger problems; smaller tiles for small H/N
        BLOCK_N = 128 if N >= 128 else (64 if N >= 64 else 32)
        BLOCK_H = 128 if H >= 128 else (64 if H >= 64 else 32)

        # Launch configuration
        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(H, BLOCK_H))
        num_warps = 4
        num_stages = 2

        scatter_add_tiles_kernel[grid](
            output, expert_outputs, token_indices,
            N, H,
            BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=num_stages,
        )

        return output


def run(*args):
    return ModelNew()(*args)
