import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_add_full_row_kernel(
    out_fp32_ptr,            # *float32, [M, H]
    token_indices_ptr,       # *int32, [N]
    expert_outputs_ptr,      # *float32, [N, H]
    N: tl.constexpr,         # number of updates
    H: tl.constexpr,         # hidden size
    T: tl.constexpr,         # updates per program
    BLOCK_H: tl.constexpr,   # should be H for full-row
):
    pid = tl.program_id(axis=0)
    # Each program processes up to T updates
    base = pid * T
    offs = tl.arange(0, BLOCK_H)  # vector covering the entire row

    # Process lanes 0..T-1 (masked)
    for k in range(T):
        pos = base + k
        mask_pos = pos < N
        # Load index for this update
        idx = tl.load(token_indices_ptr + pos, mask=mask_pos, other=0)
        # Load vector of this row
        v = tl.load(expert_outputs_ptr + pos * H + offs, mask=mask_pos, other=0.0)
        # Atomic add into the corresponding row
        tl.atomic_add(out_fp32_ptr + idx * H + offs, v, mask=mask_pos)


@triton.jit
def _scatter_add_chunked_kernel(
    out_fp32_ptr,            # *float32, [M, H]
    token_indices_ptr,       # *int32, [N]
    expert_outputs_ptr,      # *float32, [N, H]
    N,                       # number of updates (int)
    H,                       # hidden size (int)
    T: tl.constexpr,         # updates per program
    BLOCK_H: tl.constexpr,   # chunk size along hidden dim
):
    pid = tl.program_id(axis=0)
    base = pid * T
    # Vector of hidden offsets for a chunk
    h_offsets = tl.arange(0, BLOCK_H)

    # Process lanes 0..T-1 (masked)
    for k in range(T):
        pos = base + k
        mask_pos = pos < N
        # Load index for this update
        idx = tl.load(token_indices_ptr + pos, mask=mask_pos, other=0)
        # Iterate across the hidden dimension in chunks
        # num_blocks is a compile-time constant derived from H and BLOCK_H
        num_blocks = (H + BLOCK_H - 1) // BLOCK_H
        for blk in range(num_blocks):
            h = blk * BLOCK_H
            offs = h + h_offsets
            mask = (offs < H) & mask_pos
            v = tl.load(expert_outputs_ptr + pos * H + offs, mask=mask, other=0.0)
            tl.atomic_add(out_fp32_ptr + idx * H + offs, v, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add:
          output[token_indices[i]] += expert_outputs[i, :]
        Accumulation is performed in float32 via atomic_add, then cast back to bfloat16.
        """
        # Ensure device and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA"
        M = final_hidden_states.shape[0]
        N = expert_outputs.shape[0]
        H = final_hidden_states.shape[1]
        # Make sure shapes match
        assert token_indices.shape[0] == N, "token_indices length must equal num_selected_tokens"
        assert expert_outputs.shape[1] == H, "expert_outputs second dimension must equal hidden_size"

        # Prepare fp32 output for atomic accumulation
        out_fp32 = final_hidden_states.clone().to(torch.float32)
        # Ensure tensors are contiguous
        out_fp32 = out_fp32.contiguous()
        expert_outputs_fp32 = expert_outputs.to(torch.float32).contiguous()
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        # Heuristics for BLOCK_H and num_warps
        if H <= 1024:
            BLOCK_H = 1024
            num_warps = 8
            T = 4  # process 4 updates per program
            grid = (triton.cdiv(N, T),)
            _scatter_add_full_row_kernel[grid](
                out_fp32, token_indices_i32, expert_outputs_fp32, N, H, T, BLOCK_H,
                num_warps=num_warps, num_stages=2,
            )
        else:
            # For larger hidden sizes, process in chunks
            # Choose BLOCK_H based on H
            if H >= 2048:
                BLOCK_H = 1024
                num_warps = 8
            elif H >= 512:
                BLOCK_H = 512
                num_warps = 4
            else:
                BLOCK_H = 256
                num_warps = 2
            T = 4
            grid = (triton.cdiv(N, T),)
            _scatter_add_chunked_kernel[grid](
                out_fp32, token_indices_i32, expert_outputs_fp32, N, H, T, BLOCK_H,
                num_warps=num_warps, num_stages=2,
            )

        # Cast back to bfloat16 to match original API
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
