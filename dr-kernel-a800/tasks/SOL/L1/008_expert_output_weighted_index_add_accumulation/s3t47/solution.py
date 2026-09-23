import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_2d_chunks(
    out_ptr,        # *float32, shape [B, H], B = batch_seq_len
    idx_ptr,        # *int32,   shape [N]
    src_ptr,        # *float32, shape [N, H]
    N,              # int32: number of updates (len(token_indices))
    B,              # int32: number of tokens (batch_seq_len), not used directly
    H,              # int32: hidden_size
    BLOCK_H: tl.constexpr,  # vector width across hidden dim
):
    # 2D grid: pid0 over tokens, pid1 over hidden chunks
    pid_i = tl.program_id(0)  # token index
    pid_c = tl.program_id(1)  # chunk index over hidden dimension

    # Mask for valid token
    mask_i = pid_i < N
    # Compute hidden offsets for this chunk
    h_start = pid_c * BLOCK_H
    offs = h_start + tl.arange(0, BLOCK_H)
    mask_h = offs < H

    # Load target token index
    idx = tl.load(idx_ptr + pid_i, mask=mask_i, other=0)  # int32 scalar
    # Compute row base offset in output
    out_row_base = idx * H

    # Compute source row base offset
    src_row_base = pid_i * H

    # Load source chunk (vector), masked for H tail
    v = tl.load(src_ptr + src_row_base + offs, mask=mask_i & mask_h, other=0.0)

    # Atomic add into output row chunk
    tl.atomic_add(out_ptr + out_row_base + offs, v, mask=mask_i & mask_h)


class ModelNew(torch.nn.Module):
    def forward(
        final_hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        # Device checks
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "CUDA tensors required."
        device = final_hidden_states.device

        # Shapes
        B = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]

        # Prepare buffers
        out_fp32 = final_hidden_states.clone().to(torch.float32).contiguous()
        src_fp32 = expert_outputs.contiguous().to(torch.float32)
        idx_i32 = token_indices.to(torch.int32).contiguous()

        # Heuristics for BLOCK_H and num_warps
        if H >= 1024:
            BLOCK_H = 1024
            num_warps = 8
        elif H >= 512:
            BLOCK_H = 512
            num_warps = 4
        else:
            BLOCK_H = 256
            num_warps = 2

        # Grid: (N tokens, ceil_div(H, BLOCK_H) chunks)
        grid = (N, triton.cdiv(H, BLOCK_H))

        # Launch 2D tiled atomic add across hidden dimension
        scatter_add_2d_chunks[grid](
            out_fp32,
            idx_i32,
            src_fp32,
            N,
            B,
            H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=4,
        )

        # Cast back to bfloat16 to match original interface
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
