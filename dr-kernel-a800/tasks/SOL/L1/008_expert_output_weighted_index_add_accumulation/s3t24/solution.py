import torch
import triton
import triton.language as tl


def _next_power_of_two(x: int) -> int:
    # Returns the next power-of-two >= x, capped at 1024
    if x <= 1:
        return 1
    p = 1
    while p < x and p < 1024:
        p <<= 1
    return p


@triton.jit
def scatter_add_single_update_kernel(
    out_ptr,          # *float32, shape [M, H]
    expert_ptr,       # *float32, shape [N, H]
    idx_ptr,          # *int32,   shape [N]
    N: tl.constexpr,  # int
    M: tl.constexpr,  # int
    H: tl.constexpr,  # int
    BLOCK_H: tl.constexpr,  # int, vector width >= H
):
    # One program per update
    pid = tl.program_id(0)
    # If pid >= N, nothing to do
    if pid >= N:
        return

    # Load destination row index
    idx = tl.load(idx_ptr + pid)  # int32
    if idx < 0 or idx >= M:
        return  # defensive, though token_indices are valid in provided inputs

    # Compute base pointers for this update and destination row
    # expert_ptr is row-major: row i has offset i * H
    base_expert = pid * H
    base_out = idx * H

    # Load the entire vector v = expert_outputs[pid, :] as float32
    offs = tl.arange(0, BLOCK_H)  # [BLOCK_H]
    mask = offs < H
    v = tl.load(expert_ptr + base_expert + offs, mask=mask, other=0.0)  # [BLOCK_H], float32

    # Atomic add v into out[base_out + offs]
    # out_ptr is row-major: out[row, :] base at row * H
    tl.atomic_add(out_ptr + base_out + offs, v, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        """
        Triton implementation of:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        where final_hidden_states: [batch_seq_len, hidden_size], bfloat16
              expert_outputs: [num_selected_tokens, hidden_size], bfloat16
              token_indices: [num_selected_tokens], long
        Returns output in bfloat16, with duplicates summed.
        """
        # Ensure all tensors are on CUDA
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA"

        # Shapes
        M = final_hidden_states.shape[0]
        N = expert_outputs.shape[0]
        H = expert_outputs.shape[1]

        # Accumulate in float32 for correctness and performance
        out_fp32 = final_hidden_states.to(torch.float32).clone()  # [M, H], float32

        # Prepare inputs for Triton
        expert_fp32 = expert_outputs.to(torch.float32)  # [N, H], float32
        token_int = token_indices.to(torch.int32)      # [N], int32

        # Ensure contiguity
        out_fp32 = out_fp32.contiguous()
        expert_fp32 = expert_fp32.contiguous()
        token_int = token_int.contiguous()

        # Select BLOCK_H as next power-of-two >= H, capped at 1024
        BLOCK_H = _next_power_of_two(H)

        # Launch Triton kernel: one program per update
        grid = (N,)

        # Choose num_warps based on BLOCK_H; 4 or 8 are good defaults
        num_warps = 4 if BLOCK_H <= 256 else 8

        scatter_add_single_update_kernel[grid](
            out_fp32, expert_fp32, token_int,
            N, M, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
        )

        # Cast back to bfloat16 to match original
        result = out_fp32.to(torch.bfloat16)
        return result


def run(*args):
    return ModelNew()(*args)
