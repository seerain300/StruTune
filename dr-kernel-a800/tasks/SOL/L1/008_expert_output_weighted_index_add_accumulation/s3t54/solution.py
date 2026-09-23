import torch
import triton
import triton.language as tl

@triton.jit
def scatter_add_rows_fp32_kernel(
    out_ptr,             # *fp32, shape [batch_seq_len, H]
    token_indices_ptr,   # *int32, shape [N]
    expert_outputs_ptr,  # *fp32, shape [N, H]
    N,                   # int: number of updates
    H,                   # int: hidden_size
    BLOCK_H: tl.constexpr,  # compile-time constant chunk size
):
    # One program per update i
    i = tl.program_id(axis=0)
    if i >= N:
        return

    # Load index for this update
    idx = tl.load(token_indices_ptr + i)
    # Guard against any unexpected negative indices (shouldn't happen with provided data)
    # Triton will handle out-of-bounds via masks in loads/stores.

    # Iterate over hidden dimension in chunks of BLOCK_H
    # Note: we use a Python for-loop with range in Triton; H is a runtime int,
    # but BLOCK_H is constexpr. Triton will generate the loop with unrolling if possible.
    for h in range(0, H, BLOCK_H):
        offs = h + tl.arange(0, BLOCK_H)
        mask = offs < H

        # Load the vector chunk from expert_outputs for this update i
        v = tl.load(expert_outputs_ptr + i * H + offs, mask=mask, other=0.0)  # fp32 vector

        # Compute output pointer for the selected row and do atomic add
        out_row_ptr = out_ptr + idx * H + offs
        tl.atomic_add(out_row_ptr, v, mask=mask)


def _choose_block_h_and_warps(H: int):
    # Choose BLOCK_H as the largest power-of-two not exceeding H, capped at 1024.
    if H >= 1024:
        BLOCK_H = 1024
        num_warps = 8
    elif H >= 512:
        BLOCK_H = 512
        num_warps = 8
    elif H >= 256:
        BLOCK_H = 256
        num_warps = 4
    else:
        # For very small H, use 128 to avoid overly small vectors
        BLOCK_H = 128
        num_warps = 2
    return BLOCK_H, num_warps


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure tensors are on the same CUDA device and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA"
        device = final_hidden_states.device
        N = expert_outputs.shape[0]
        H = final_hidden_states.shape[1]

        # Prepare output as float32 for atomic adds (bf16 atomics not supported)
        out_fp32 = final_hidden_states.clone().to(torch.float32)

        # Ensure expert_outputs is float32 contiguous
        expert_outputs_fp32 = expert_outputs.to(torch.float32).contiguous()

        # Ensure token_indices is int32 contiguous
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        # Choose BLOCK_H and num_warps based on H
        BLOCK_H, num_warps = _choose_block_h_and_warps(H)

        # Launch one program per update
        grid = (N,)

        scatter_add_rows_fp32_kernel[grid](
            out_fp32,
            token_indices_i32,
            expert_outputs_fp32,
            N,
            H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=2,
        )

        # Cast back to bfloat16 to match original API
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
