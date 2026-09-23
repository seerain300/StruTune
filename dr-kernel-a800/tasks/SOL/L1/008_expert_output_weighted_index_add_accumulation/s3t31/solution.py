import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_chunks_kernel(
    out_ptr,          # *fp32, shape [M, H]
    token_indices_ptr,# *i32,  shape [N]
    expert_ptr,       # *fp32, shape [N, H]
    M: tl.constexpr,  # number of rows in output
    H: tl.constexpr,  # hidden size
    N: tl.constexpr,  # number of updates
    BLOCK_H: tl.constexpr,
):
    # One program per update
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Load target row index
    idx = tl.load(token_indices_ptr + pid)
    if idx >= M:
        return

    # Base pointer for this output row
    out_row_ptr = out_ptr + idx * H

    # Vectorized processing over hidden dimension in chunks of BLOCK_H
    for h in range(0, H, BLOCK_H):
        offs = h + tl.arange(0, BLOCK_H)
        mask = offs < H
        # Load expert vector chunk
        v = tl.load(expert_ptr + pid * H + offs, mask=mask, other=0.0)
        # Atomic add into the output row
        tl.atomic_add(out_row_ptr + offs, v, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        final_hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        token_indices: torch.Tensor,
    ):
        """
        Triton-accelerated scatter-add:
        output = final_hidden_states.clone()
        output.index_add_(dim=0, index=token_indices, source=expert_outputs)

        Returns: output with bfloat16 dtype, matching the original API.
        """
        # Ensure tensors are on CUDA
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA device"
        device = final_hidden_states.device

        # Accumulate in float32 for atomic_add support
        out_fp32 = final_hidden_states.contiguous().to(torch.float32).clone()

        # Dimensions
        M = out_fp32.shape[0]  # batch_seq_len
        H = out_fp32.shape[1]
        N = expert_outputs.shape[0]
        assert token_indices.shape[0] == N

        # Triton prefers int32 indices; ensure contiguous and proper dtypes
        token_indices_i32 = token_indices.contiguous().to(torch.int32)
        expert_outputs_fp32 = expert_outputs.contiguous().to(torch.float32)

        # Choose BLOCK_H and num_warps based on H
        if H >= 1024:
            BLOCK_H = 1024
            num_warps = 8
        elif H >= 512:
            BLOCK_H = 512
            num_warps = 4
        else:
            BLOCK_H = 256
            num_warps = 2

        # Launch kernel: one program per update
        grid = (N,)
        scatter_add_row_chunks_kernel[grid](
            out_fp32,
            token_indices_i32,
            expert_outputs_fp32,
            M, H, N,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=2,
        )

        # Cast back to bfloat16 to match original API
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
