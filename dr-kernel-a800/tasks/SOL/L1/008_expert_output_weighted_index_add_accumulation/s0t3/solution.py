import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_blocks_kernel(
    output_ptr,           # *ptr to output [M, H] (bf16)
    expert_outputs_ptr,   # *ptr to expert_outputs [N, H] (bf16)
    token_indices_ptr,    # *ptr to token_indices [N] (int64)
    N: tl.constexpr,      # number of source rows
    H: tl.constexpr,      # hidden size
    BLOCK_N: tl.constexpr,  # tile size along N
    BLOCK_H: tl.constexpr   # tile size along H
):
    # 2D grid over (N-blocks, H-blocks)
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    h_start = pid_h * BLOCK_H

    # Offsets within this tile
    n_offsets = n_start + tl.arange(0, BLOCK_N)             # [BLOCK_N]
    h_offsets = h_start + tl.arange(0, BLOCK_H)             # [BLOCK_H]

    # Masks for bounds
    mask_n = n_offsets < N
    mask_h = h_offsets < H

    # Load target indices for each n in this block
    idx64 = tl.load(token_indices_ptr + n_offsets, mask=mask_n, other=0)  # int64 vector of size BLOCK_N

    # Compute 2D pointers into output and expert_outputs
    # Output: output[idx, h] -> address = idx * H + h
    out_ptrs = idx64[:, None] * H + h_offsets[None, :]             # shape [BLOCK_N, BLOCK_H]
    # Expert: expert_outputs[n, h] -> address = n * H + h
    exp_ptrs = n_offsets[:, None] * H + h_offsets[None, :]         # shape [BLOCK_N, BLOCK_H]

    # Combined mask for the 2D tile
    mask = mask_n[:, None] & mask_h[None, :]                       # shape [BLOCK_N, BLOCK_H]

    # Load values from expert_outputs for this tile
    vals = tl.load(expert_outputs_ptr + exp_ptrs, mask=mask, other=0.0)  # [BLOCK_N, BLOCK_H]

    # Atomically add into output
    tl.atomic_add(output_ptr + out_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add along dim=0:
          output = final_hidden_states.clone()
          output[token_indices[i]] += expert_outputs[i] for i in [0, expert_outputs.size(0))
        """
        # Ensure CUDA tensors and dtypes
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Triton kernels require CUDA tensors"
        assert final_hidden_states.dtype == expert_outputs.dtype, "final_hidden_states and expert_outputs must have the same dtype (e.g., bfloat16)"
        assert token_indices.dtype in (torch.int64, torch.int32), "token_indices must be integer type (int64/int32)"

        # Initialize output as clone of final_hidden_states (fresh buffer)
        output = final_hidden_states.clone()

        # Shapes
        N = expert_outputs.shape[0]
        H = expert_outputs.shape[1]

        # Heuristic selection for BLOCK_H and launch params
        # Choose larger H-block for larger H to improve vectorization
        if H >= 2048:
            BLOCK_H = 256
            num_warps = 8
            num_stages = 3
        elif H >= 512:
            BLOCK_H = 128
            num_warps = 4
            num_stages = 2
        elif H >= 128:
            BLOCK_H = 64
            num_warps = 4
            num_stages = 2
        else:
            BLOCK_H = 32
            num_warps = 2
            num_stages = 2

        # For N, choose a moderate BLOCK_N to increase parallelism
        BLOCK_N = 64 if N >= 64 else 32

        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(H, BLOCK_H))

        # Launch Triton kernel
        scatter_add_blocks_kernel[grid](
            output, expert_outputs, token_indices,
            N, H,
            BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=num_stages,
        )

        return output


def run(*args):
    return ModelNew()(*args)
