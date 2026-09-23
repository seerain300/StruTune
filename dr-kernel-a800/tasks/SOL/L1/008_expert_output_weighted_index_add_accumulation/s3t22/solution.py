import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_vec_atomic_fp32(
    out_ptr,         # *float32, [M, H]
    idx_ptr,         # *int32,   [N]
    exp_ptr,         # *float32, [N, H]
    M: tl.int32,     # rows in out
    H: tl.int32,     # cols
    N: tl.int32,     # number of updates
    BLOCK_H: tl.constexpr,
):
    # One program per update
    i = tl.program_id(0)
    if i >= N:
        return

    # Load destination row index for this update
    idx = tl.load(idx_ptr + i)

    # Process the hidden dimension in chunks of BLOCK_H
    offs = tl.arange(0, BLOCK_H)
    for h in range(0, H, BLOCK_H):
        col = h + offs
        mask = col < H
        # Load chunk from expert_outputs[i, :]
        v = tl.load(exp_ptr + i * H + col, mask=mask, other=0.0)  # [BLOCK_H], fp32
        # Atomic add into out_fp32[idx, col]
        tl.atomic_add(out_ptr + idx * H + col, v, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized scatter-add along rows:
        out = torch.zeros_like(final_hidden_states, dtype=float32)
        out[token_indices[i]] += expert_outputs[i] for each i
        Return out cast to bfloat16 to match the original forward's output dtype.
        """
        # Ensure tensors are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All inputs must be on CUDA for Triton kernels."
        final_hs = final_hidden_states.contiguous()
        exp = expert_outputs.contiguous()
        tok = token_indices.contiguous()

        M, H = final_hs.shape
        N = exp.shape[0]

        # Prepare fp32 accumulation buffer (must start at zeros, index_add semantics require accumulation)
        out_fp32 = torch.zeros((M, H), dtype=torch.float32, device=final_hs.device)

        # Convert expert_outputs to fp32 (original is bfloat16)
        exp_fp32 = exp.to(torch.float32)

        # Convert token_indices to int32
        tok_i32 = tok.to(torch.int32)

        # Choose BLOCK_H based on H for better vectorization
        if H >= 1024:
            BLOCK_H = 1024
            num_warps = 4
        elif H >= 512:
            BLOCK_H = 512
            num_warps = 4
        elif H >= 256:
            BLOCK_H = 256
            num_warps = 4
        else:
            BLOCK_H = 128
            num_warps = 4

        # Launch one program per update
        grid = (N,)
        scatter_add_rows_vec_atomic_fp32[grid](
            out_fp32, tok_i32, exp_fp32,
            M, H, N,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=2,
        )

        # Cast back to bfloat16 to match original dtype
        out_bf16 = out_fp32.to(torch.bfloat16)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
