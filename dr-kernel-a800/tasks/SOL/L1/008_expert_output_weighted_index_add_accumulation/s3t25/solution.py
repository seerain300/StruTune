import torch
import triton
import triton.language as tl


@triton.jit
def atomic_add_rows_chunked(
    out_ptr,        # *fp32, shape [M, H]
    expert_ptr,     # *fp32, shape [N, H]
    indices_ptr,    # *i32,  shape [N]
    M, N, H,        # int32 scalars
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per update
    if pid >= N:
        return

    # Load destination row index (int32)
    idx = tl.load(indices_ptr + pid)
    if idx < 0 or idx >= M:
        return  # defensive check

    # Iterate over hidden dimension in chunks of BLOCK_H
    for h in range(0, H, BLOCK_H):
        offs = h + tl.arange(0, BLOCK_H)
        mask = offs < H

        # Load the chunk of expert_outputs for this update (fp32), masked tail elements as 0
        v = tl.load(expert_ptr + pid * H + offs, mask=mask, other=0.0)

        # Compute output pointers for the chunk at row idx
        out_ptrs = out_ptr + idx * H + offs

        # Perform a single atomic add per block
        tl.atomic_add(out_ptrs, v, mask=mask)


def _choose_block_h(hidden_size: int):
    # Choose a power-of-two block size up to 1024, not exceeding hidden_size.
    if hidden_size >= 1024:
        return 1024, 8  # BLOCK_H, num_warps
    elif hidden_size >= 512:
        return 512, 8
    else:
        return 256, 4


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add that accumulates expert_outputs into
        final_hidden_states at positions specified by token_indices, using
        fp32 atomic adds in chunks for performance and correctness, then
        casting back to bfloat16.
        """
        # Ensure device and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA device for Triton kernels."
        M = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]

        # Prepare fp32 accumulation buffer (clone initial state in fp32)
        out_fp32 = final_hidden_states.float().clone()

        # Ensure expert_outputs is float32 and contiguous
        # The original expert_outputs are bfloat16; we convert to float32 for accumulation.
        expert_outputs_fp32 = expert_outputs.float().contiguous()

        # Ensure indices are int32 for Triton
        indices_i32 = token_indices.to(torch.int32).contiguous()

        # Launch Triton kernel: one program per update
        BLOCK_H, num_warps = _choose_block_h(H)
        grid = (N,)
        atomic_add_rows_chunked[grid](
            out_fp32, expert_outputs_fp32, indices_i32,
            M, N, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
        )

        # Cast back to bfloat16 to match original interface
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
