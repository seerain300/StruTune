import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_chunked_kernel(
    out_ptr,            # *bfloat16
    expert_ptr,         # *bfloat16
    indices_ptr,        # *int32
    N,                  # int32: number of rows (batch_seq_len)
    H,                  # int32: hidden_size
    BLOCK_H: tl.constexpr,  # chunk size along hidden dimension
):
    # One Triton program per row
    row_id = tl.program_id(0)
    if row_id >= N:
        return

    # Destination row index (int32)
    row_idx = tl.load(indices_ptr + row_id)

    # Process hidden dimension in chunks of BLOCK_H
    # For each chunk, iterate scalarly over j to keep Triton pointer arithmetic simple and safe.
    for start in range(0, H, BLOCK_H):
        # Scalar loop over the chunk
        for j in range(0, BLOCK_H):
            idx = start + j
            # Mask to avoid out-of-bounds when H is not divisible by BLOCK_H
            if idx < H:
                src_ptr = expert_ptr + row_id * H + idx
                dst_ptr = out_ptr + row_idx * H + idx
                val = tl.load(src_ptr)
                tl.atomic_add(dst_ptr, val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        # Match the original behavior: clone the input buffer
        out = final_hidden_states.clone()

        # Ensure CUDA and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA"
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()

        # Triton expects int32 indices for pointer arithmetic
        indices_i32 = token_indices.to(torch.int32)

        N = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]

        # Launch: one program per row
        grid = (N,)
        # Choose a modest chunk size; 128 is a good default for typical hidden sizes.
        # If H is small, the loop just runs once.
        scatter_add_rows_chunked_kernel[grid](
            out,
            expert_outputs,
            indices_i32,
            N,
            H,
            BLOCK_H=128,
            num_warps=1,     # keep it simple; per-row work is not huge
            num_stages=1,
        )
        return out


def run(*args):
    return ModelNew()(*args)
