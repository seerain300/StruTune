import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_add_chunked_one_update(
    out_ptr,            # *float32, shape [B, H], B = batch_seq_len
    idx_ptr,            # *int32,   shape [N]
    src_ptr,            # *float32, shape [N, H] (row i maps to idx[i])
    N,                  # int32: number of updates
    B,                  # int32: number of tokens (batch_seq_len)
    H,                  # int32: hidden_size
    BLOCK_H: tl.constexpr,  # vector width across hidden dim
):
    pid = tl.program_id(0)  # each program handles exactly one update
    # Bounds check
    if pid >= N:
        return
    # Load the target token index (int32)
    idx = tl.load(idx_ptr + pid)
    # Base offset for the row in out_ptr
    out_base = idx * H
    # Iterate over hidden dimension in BLOCK_H chunks
    for h_start in range(0, H, BLOCK_H):
        cols = tl.arange(0, BLOCK_H)
        h_offsets = h_start + cols
        mask_cols = h_offsets < H
        # Load the source chunk as float32
        src_chunk = tl.load(src_ptr + pid * H + h_offsets, mask=mask_cols, other=0.0)
        # Atomic add the chunk into out at the target row
        tl.atomic_add(out_ptr + out_base + h_offsets, src_chunk, mask=mask_cols)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        final_hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        token_indices: torch.Tensor,
    ):
        # Ensure all inputs are on the same CUDA device and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA."
        device = final_hidden_states.device

        # Convert token_indices to int32 for Triton
        idx_i32 = token_indices.to(torch.int32)

        # Prepare output buffer in float32 for atomic accumulation, and contiguous buffers
        B = final_hidden_states.shape[0]  # batch_seq_len
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]

        # Clone and cast to float32 for accumulation
        out_fp32 = final_hidden_states.to(torch.float32).contiguous()
        src_fp32 = expert_outputs.to(torch.float32).contiguous()

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

        # One program per update
        grid = (N,)

        _scatter_add_chunked_one_update[grid](
            out_fp32, idx_i32, src_fp32,
            N, B, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=2,
        )

        # Cast back to bfloat16 to match original API
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
