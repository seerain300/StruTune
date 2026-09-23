import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_chunked_kernel(
    out_ptr,            # *float32 [B, H], row stride = H
    indices_ptr,        # *int32 [N]
    vals_ptr,           # *float32 [N, H], row stride = H, col stride = 1
    N,                  # int32
    H,                  # int32
    out_stride0,        # int32
    vals_stride0,       # int32
    vals_stride1,       # int32
    T: tl.constexpr,    # number of updates per program
    BLOCK_H: tl.constexpr,  # chunk size for hidden dimension
):
    # Each program handles up to T updates
    pid = tl.program_id(axis=0)
    k = tl.arange(0, T)                      # vector [T]
    pos = pid * T + k                        # positions in [0, N)
    mask_pos = pos < N

    # Load indices for these positions
    idx = tl.load(indices_ptr + pos, mask=mask_pos, other=0)  # int32
    out_row_ptr = out_ptr + idx * out_stride0
    vals_row_ptr = vals_ptr + pos * vals_stride0

    # Iterate over hidden dimension in chunks of BLOCK_H
    num_chunks = (H + BLOCK_H - 1) // BLOCK_H
    for chunk in range(0, num_chunks):
        h_start = chunk * BLOCK_H
        h_offsets = h_start + tl.arange(0, BLOCK_H)            # [BLOCK_H]
        mask_cols = h_offsets < H
        mask = mask_pos[:, None] & mask_cols[None, :]
        # Load vals chunk for each of the T updates: shape [T, BLOCK_H]
        v = tl.load(vals_row_ptr[:, None] + h_offsets[None, :] * vals_stride1,
                    mask=mask, other=0.0)
        # Atomic add into output (float32 accumulation)
        tl.atomic_add(out_row_ptr[:, None], v, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized scatter-add equivalent to:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We accumulate in float32 for performance and numerical stability, then cast to bfloat16.
        """
        # Ensure CUDA and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All inputs must be on CUDA device"
        out_bf16 = final_hidden_states  # keep original for returning bfloat16
        out_fp32 = final_hidden_states.to(torch.float32).clone()
        vals_fp32 = expert_outputs.to(torch.float32)
        indices_i32 = token_indices.to(torch.int32)

        B = out_fp32.shape[0]
        H = out_fp32.shape[1]
        N = vals_fp32.shape[0]
        assert vals_fp32.shape == (N, H), "expert_outputs must have shape (N, H)"
        assert indices_i32.shape == (N,), "token_indices must have shape (N,)"

        # Choose kernel parameters
        T = 4  # number of updates per program
        BLOCK_H = 1024 if H >= 1024 else (512 if H >= 512 else 256)
        num_warps = 8 if BLOCK_H >= 512 else 4
        grid = (triton.cdiv(N, T),)

        scatter_add_chunked_kernel[grid](
            out_fp32, indices_i32, vals_fp32,
            N, H,
            out_fp32.stride(0), vals_fp32.stride(0), vals_fp32.stride(1),
            T=T, BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=2,
        )

        # Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
