import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_chunked_bf16_kernel(
    out_ptr,            # *float32, shape [B, H]
    indices_ptr,        # *int32, shape [N]
    vals_ptr,           # *float32, shape [N, H]
    N,                  # number of updates (int)
    H,                  # hidden size (int)
    BLOCK_H: tl.constexpr,
):
    # One program per update
    pid = tl.program_id(axis=0)
    # Bounds check: if grid is larger than N, mask out
    if pid >= N:
        return

    # Load the target index for this update
    idx = tl.load(indices_ptr + pid)  # int32

    # Iterate over hidden dimension in chunks of BLOCK_H
    # Note: Triton supports range over Python integers; H is passed as int.
    for h in range(0, H, BLOCK_H):
        h_offsets = h + tl.arange(0, BLOCK_H)
        mask = h_offsets < H

        # Load the vector of values for this chunk (float32)
        vals_vec = tl.load(vals_ptr + pid * H + h_offsets, mask=mask, other=0.0)

        # Compute output pointer for this row and chunk
        out_ptrs = out_ptr + idx * H + h_offsets

        # Atomic add the chunk into the output (float32)
        tl.atomic_add(out_ptrs, vals_vec, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA device."
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Prepare buffers:
        # - out in float32 for atomic accumulation
        out_fp32 = final_hidden_states.to(torch.float32).clone()
        # - vals (expert_outputs) in float32 for kernel
        vals_fp32 = expert_outputs.to(torch.float32)
        # - indices as int32 for Triton
        indices_i32 = token_indices.to(torch.int32)

        B = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]

        # Choose BLOCK_H and num_warps heuristically based on H
        if H >= 1024:
            BLOCK_H = 1024
            num_warps = 8
        elif H >= 512:
            BLOCK_H = 512
            num_warps = 4
        else:
            BLOCK_H = 256
            num_warps = 2

        # Launch one program per update
        grid = (N,)

        scatter_add_chunked_bf16_kernel[grid](
            out_fp32, indices_i32, vals_fp32, N, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=2,
        )

        # Return in bfloat16 to match original API expectations
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
