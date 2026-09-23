import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_multi_update_kernel(
    out_ptr,            # *float32, shape [M, H]
    token_indices_ptr,  # *int32, shape [N]
    expert_outputs_ptr, # *float32, shape [N, H]
    N,                  # int32, total number of updates
    H,                  # int32, hidden size
    T: tl.constexpr,    # updates per program (compile-time constant)
    BLOCK_H: tl.constexpr,  # block size along hidden dimension
):
    # Each program handles T updates
    pid = tl.program_id(axis=0)
    start = pid * T

    # Vectorized lanes for the T updates
    k = tl.arange(0, T)            # [0..T-1]
    pos = start + k                # positions in [0..N)
    mask_k = pos < N               # mask for valid lanes

    # Load token indices for each of the T lanes
    idx = tl.load(token_indices_ptr + pos, mask=mask_k, other=0)  # int32

    # Iterate over hidden dimension in chunks
    for h in range(0, H, BLOCK_H):
        offs = h + tl.arange(0, BLOCK_H)      # column offsets
        mask_offs = offs < H                   # mask for last block

        # 2D mask for valid lanes and valid columns
        mask = mask_k[:, None] & mask_offs[None, :]

        # Load v_chunk for each lane: shape [T, BLOCK_H]
        v = tl.load(
            expert_outputs_ptr + pos[:, None] * H + offs[None, :],
            mask=mask,
            other=0.0,
        )  # float32

        # Atomic add into output
        tl.atomic_add(out_ptr + idx[:, None] * H + offs[None, :], v, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Fallback to PyTorch if not on CUDA (safety)
        if (not final_hidden_states.is_cuda) or (not expert_outputs.is_cuda) or (not token_indices.is_cuda):
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices, expert_outputs.to(output.dtype))
            return output

        # Shapes
        M = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]

        # Output buffer in float32 for atomic accumulation
        out_fp32 = final_hidden_states.clone().to(torch.float32)

        # Ensure inputs are contiguous and dtype correct
        token_indices_i32 = token_indices.to(torch.int32).contiguous()
        expert_outputs_fp32 = expert_outputs.contiguous().to(torch.float32)

        # Heuristics for BLOCK_H and num_warps based on H
        if H >= 2048:
            BLOCK_H = 1024
            num_warps = 8
        elif H >= 1024:
            BLOCK_H = 1024
            num_warps = 8
        elif H >= 512:
            BLOCK_H = 512
            num_warps = 4
        elif H >= 256:
            BLOCK_H = 256
            num_warps = 2
        elif H >= 128:
            BLOCK_H = 128
            num_warps = 2
        elif H >= 64:
            BLOCK_H = 64
            num_warps = 1
        elif H >= 32:
            BLOCK_H = 32
            num_warps = 1
        elif H >= 16:
            BLOCK_H = 16
            num_warps = 1
        elif H >= 8:
            BLOCK_H = 8
            num_warps = 1
        elif H >= 4:
            BLOCK_H = 4
            num_warps = 1
        else:
            BLOCK_H = 1
            num_warps = 1

        # Process multiple updates per program: T=4 is a good balance
        T = 4

        # Grid size: number of programs, each handles T updates
        grid = (triton.cdiv(N, T),)

        scatter_add_rows_multi_update_kernel[grid](
            out_fp32,
            token_indices_i32,
            expert_outputs_fp32,
            N,
            H,
            T=T,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=2,
        )

        # Cast back to bfloat16 to match original API
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
