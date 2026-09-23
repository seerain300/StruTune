import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_chunks(
    out_ptr,            # *float32, shape [B, H], B = batch_seq_len
    idx_ptr,            # *int32,   shape [N]
    src_ptr,            # *float32, shape [N, H]
    N,                  # int32: number of updates
    B,                  # int32: number of tokens (batch_seq_len)
    H,                  # int32: hidden_size
    BLOCK_H: tl.constexpr,  # vector width across hidden dim
):
    # One program per update
    pid = tl.program_id(0)
    # Mask for valid update
    if pid >= N:
        return
    # Load target token index (int32)
    idx = tl.load(idx_ptr + pid)
    # Base offset for the row in out_ptr
    out_base = idx * H
    # Iterate over hidden dimension in chunks of BLOCK_H
    for h in range(0, H, BLOCK_H):
        offs = tl.arange(0, BLOCK_H)
        mask = (h + offs) < H
        # Load source vector (float32) for this update and chunk
        src_row = tl.load(src_ptr + pid * H + h + offs, mask=mask, other=0.0)
        # Atomic add into the output row
        tl.atomic_add(out_ptr + out_base + h + offs, src_row, mask=mask)


def _choose_block_and_warps(H: int):
    # Heuristics for BLOCK_H and num_warps
    if H >= 1024:
        return 1024, 8
    elif H >= 512:
        return 512, 4
    else:
        return 256, 2


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add:
          output[token_indices[i]] += expert_outputs[i, :]
        Returns updated final_hidden_states (cast back to bfloat16).
        """
        # Ensure CUDA device and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All inputs must be on CUDA device for Triton kernel"
        B = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]

        # Prepare dtypes: accumulate in float32 (bfloat16 atomic_add not supported)
        out_fp32 = final_hidden_states.contiguous().to(torch.float32)

        # Ensure src and idx tensors are contiguous
        src_fp32 = expert_outputs.contiguous().to(torch.float32)
        idx_i32 = token_indices.contiguous().to(torch.int32)

        # Launch Triton kernel: one program per update
        BLOCK_H, num_warps = _choose_block_and_warps(H)
        grid = (N,)
        scatter_add_row_chunks[grid](
            out_fp32, idx_i32, src_fp32, N, B, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=2,
        )

        # Cast back to bfloat16 to match original API
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
