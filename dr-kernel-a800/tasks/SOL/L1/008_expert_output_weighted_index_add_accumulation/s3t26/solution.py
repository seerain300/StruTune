import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_tokens_multi_update(
    out_ptr,              # *f32 [M, H]
    indices_ptr,          # *i32 [N]
    src_ptr,              # *f32 [N, H]
    N: tl.int32,          # number of updates
    M: tl.int32,          # number of rows (batch_seq_len)
    H: tl.int32,          # hidden_size
    BLOCK_H: tl.constexpr,  # chunk size along hidden dim
    T: tl.constexpr,       # number of updates per program (lanes)
):
    # 2D grid: axis-0 is block of updates, axis-1 is lane within the block
    pid_blk = tl.program_id(0)
    pid_lane = tl.program_id(1)

    # Base update index for this block
    base = pid_blk * T
    # Absolute update index for this lane
    i = base + pid_lane

    # Mask: valid updates within N
    lane_mask = i < N

    # Load token index (int32)
    idx = tl.load(indices_ptr + i, mask=lane_mask, other=0)

    # Iterate over hidden dimension in BLOCK_H chunks
    h = 0
    while h < H:
        offs = h + tl.arange(0, BLOCK_H)  # vector of hidden offsets
        mask_vec = offs < H               # valid positions in the row

        # Load the source vector for this update (masked)
        # Note: when lane_mask is False, we still construct a valid vector of zeros
        v = tl.load(src_ptr + i * H + offs, mask=lane_mask & mask_vec, other=0.0)

        # Atomic add into the output row
        out_row_ptr = out_ptr + idx * H
        tl.atomic_add(out_row_ptr + offs, v, mask=lane_mask & mask_vec)

        h += BLOCK_H


def _next_power_of_two_le(x: int, max_cap: int) -> int:
    # Choose largest power-of-two <= min(x, max_cap), cap at 1024
    cap = min(x, max_cap)
    if cap <= 64:
        return 64
    p = 1
    while (p << 1) <= cap:
        p <<= 1
    return p


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure all tensors are on the same CUDA device and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All inputs must be on CUDA device for Triton kernel."
        device = final_hidden_states.device

        # Shapes
        M = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]

        # Prepare output in float32 for atomic accumulation
        out_fp32 = final_hidden_states.to(torch.float32).clone()

        # Cast expert_outputs to float32 (in-place) to avoid extra allocation
        src_fp32 = expert_outputs.to(torch.float32)

        # Ensure contiguity
        out_fp32 = out_fp32.contiguous()
        src_fp32 = src_fp32.contiguous()
        token_indices = token_indices.to(torch.int32).contiguous()

        # Heuristics for BLOCK_H and T
        BLOCK_H = _next_power_of_two_le(H, 1024)  # 64, 128, 256, 512, 1024
        num_warps = 8 if BLOCK_H >= 512 else 4
        # Use T=4 as a robust default; T=8 can be tried on large N
        T = 4

        # Grid: (blocks along N, lanes within block)
        grid = (triton.cdiv(N, T), T)

        # Launch kernel
        scatter_add_tokens_multi_update[grid](
            out_fp32, token_indices, src_fp32,
            N, M, H,
            BLOCK_H=BLOCK_H,
            T=T,
            num_warps=num_warps,
            num_stages=2,
        )

        # Cast back to bfloat16 to match original interface
        result = out_fp32.to(torch.bfloat16)
        return result


def run(*args):
    return ModelNew()(*args)
