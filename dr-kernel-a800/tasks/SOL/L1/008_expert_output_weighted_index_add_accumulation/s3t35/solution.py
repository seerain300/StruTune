import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_full_row_kernel(
    out_ptr,            # *float32 [B, H]
    indices_ptr,        # *int32 [N]
    vals_ptr,           # *float32 [N, H]
    N,                  # int32: number of updates
    H: tl.constexpr,    # hidden size (constexpr for specialization)
    T: tl.constexpr,    # updates per program
):
    pid = tl.program_id(axis=0)
    start = pid * T
    lanes = tl.arange(0, T)
    pos = start + lanes
    mask_pos = pos < N

    # Load token indices
    idx = tl.load(indices_ptr + pos, mask=mask_pos, other=0).to(tl.int32)

    # Compute base offsets per row
    row_offsets = idx * H  # [T], int32

    # For each lane, do a single atomic add for the entire row
    for k in range(T):
        if pos[k] < N:
            base = pos[k] * H
            v = tl.load(vals_ptr + base + tl.arange(0, H)).to(tl.float32)  # full row
            out_row_ptr = out_ptr + row_offsets[k] + tl.arange(0, H)
            tl.atomic_add(out_row_ptr, v)


@triton.jit
def scatter_add_chunked_kernel(
    out_ptr,             # *float32 [B, H]
    indices_ptr,         # *int32 [N]
    vals_ptr,            # *float32 [N, H]
    N,                   # int32: number of updates
    H,                   # int32: hidden size (runtime)
    T: tl.constexpr,     # updates per program
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    start = pid * T
    lanes = tl.arange(0, T)
    pos = start + lanes
    mask_pos = pos < N

    # Load indices
    idx = tl.load(indices_ptr + pos, mask=mask_pos, other=0).to(tl.int32)

    # Iterate over hidden dimension in blocks
    num_blocks = (H + BLOCK_H - 1) // BLOCK_H
    for b in range(0, num_blocks):
        h_start = b * BLOCK_H
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H

        for k in range(T):
            if pos[k] < N:
                base = pos[k] * H
                v = tl.load(vals_ptr + base + h_offsets).to(tl.float32)
                out_ptrs = out_ptr + idx[k] * H + h_offsets
                tl.atomic_add(out_ptrs, v)


def _select_config(H: int):
    # Choose BLOCK_H, num_warps, and T based on H
    if H >= 4096:
        BLOCK_H = 1024
        num_warps = 8
        T = 8
    elif H >= 2048:
        BLOCK_H = 1024
        num_warps = 8
        T = 8
    elif H >= 1024:
        BLOCK_H = 512
        num_warps = 4
        T = 8
    elif H >= 512:
        BLOCK_H = 512
        num_warps = 4
        T = 8
    elif H >= 256:
        BLOCK_H = 256
        num_warps = 2
        T = 8
    elif H >= 128:
        BLOCK_H = 128
        num_warps = 2
        T = 4
    else:
        BLOCK_H = 64
        num_warps = 1
        T = 4
    return BLOCK_H, num_warps, T


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure CUDA tensors and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA"
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B, H = final_hidden_states.shape
        N = expert_outputs.shape[0]
        assert token_indices.shape[0] == N, "token_indices length must match num_selected_tokens"

        # Accumulator in float32 (bfloat16 atomic_add not supported)
        out_fp32 = final_hidden_states.to(torch.float32).clone()

        # Cast expert outputs to float32 for accumulation
        vals_fp32 = expert_outputs.to(torch.float32)

        # Triton prefers int32 indices
        indices_i32 = token_indices.to(torch.int32)

        # Kernel config
        BLOCK_H, num_warps, T = _select_config(H)

        # Grid: each program handles T updates
        grid = (triton.cdiv(N, T),)

        if H <= BLOCK_H:
            # Full-row path: minimal atomics, one add per update (vectorized full row)
            scatter_add_full_row_kernel[grid](
                out_fp32, indices_i32, vals_fp32, N, H, T,
                num_warps=num_warps, num_stages=2,
            )
        else:
            # Chunked path for larger H
            scatter_add_chunked_kernel[grid](
                out_fp32, indices_i32, vals_fp32, N, H, T, BLOCK_H,
                num_warps=num_warps, num_stages=2,
            )

        # Cast back to bfloat16 to match original API
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
