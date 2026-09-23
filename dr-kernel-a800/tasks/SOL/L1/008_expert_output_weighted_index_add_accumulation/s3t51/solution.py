import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_add_experts_vec_kernel(
    out_ptr,                # *float32, [batch_seq_len, hidden_size]
    idx_ptr,                # *int32,   [N]
    src_ptr,                # *float32, [N, hidden_size]
    N: tl.constexpr,        # number of updates (num_selected_tokens)
    H: tl.constexpr,        # hidden_size (columns)
    T: tl.constexpr,        # updates per program
    BLOCK_H: tl.constexpr,  # chunk size across hidden dimension
):
    # One program handles T updates
    pid = tl.program_id(axis=0)
    start = pid * T
    k = tl.arange(0, T)                    # vector lanes across T updates
    pos = start + k                        # [T]
    mask_pos = pos < N                     # mask for valid updates

    # Load indices for these T updates (int32)
    idx = tl.load(idx_ptr + pos, mask=mask_pos, other=0)  # [T], int32

    # Iterate over hidden dimension in chunks of BLOCK_H
    for h in range(0, H, BLOCK_H):
        h_vec = h + tl.arange(0, BLOCK_H)  # [BLOCK_H]
        mask_h = h_vec < H

        # Build 2D offsets: rows=pos[k], cols=h_vec
        pos_broadcast = pos[:, None]          # [T, 1]
        h_broadcast = h_vec[None, :]          # [1, BLOCK_H]
        src_offsets = pos_broadcast * H + h_broadcast  # [T, BLOCK_H]

        # Mask: valid position and valid hidden columns
        src_mask = mask_pos[:, None] & mask_h[None, :]  # [T, BLOCK_H]

        # Load v_chunk for all T lanes at once (float32)
        v_chunk = tl.load(src_ptr + src_offsets, mask=src_mask, other=0.0)  # [T, BLOCK_H], float32

        # Compute output offsets: out_ptr + idx[:, None] * H + h_vec[None, :]
        out_offsets = idx[:, None] * H + h_broadcast  # [T, BLOCK_H]

        # Atomic add for each lane
        tl.atomic_add(out_ptr + out_offsets, v_chunk, mask=src_mask)


def _choose_kernel_params(H: int):
    # Prefer large BLOCK_H to minimize chunk iterations for common hidden sizes
    if H >= 1024:
        BLOCK_H = 1024
        num_warps = 8
    elif H >= 512:
        BLOCK_H = 512
        num_warps = 4
    else:
        BLOCK_H = 256
        num_warps = 2
    return BLOCK_H, num_warps


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure tensors are on the same CUDA device and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA device."
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        N = expert_outputs.shape[0]
        H = final_hidden_states.shape[1]

        # Accumulate in float32 (bfloat16 atomics not supported)
        out_fp32 = final_hidden_states.clone().to(torch.float32)
        src_fp32 = expert_outputs.to(torch.float32)

        BLOCK_H, num_warps = _choose_kernel_params(H)
        # Each program handles T updates
        T = 4
        grid = (triton.cdiv(N, T),)

        _scatter_add_experts_vec_kernel[grid](
            out_fp32,
            token_indices.to(torch.int32),
            src_fp32,
            N=N,
            H=H,
            T=T,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=2,  # good default for memory-bound kernels
        )

        # Cast back to bfloat16 to match original API
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
