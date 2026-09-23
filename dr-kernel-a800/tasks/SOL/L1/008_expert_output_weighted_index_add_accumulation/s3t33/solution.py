import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,                # *fp32, shape [batch_seq_len, H]
    token_indices_ptr,      # *i32, shape [N]
    expert_outputs_ptr,     # *fp32, shape [N, H]
    N: tl.constexpr,        # number of updates (batch_seq_len * num_experts_per_tok)
    H: tl.constexpr,        # hidden size
    BLOCK_H: tl.constexpr,  # chunk size along H
):
    pid = tl.program_id(0)
    # Each program handles one update
    if pid >= N:
        return

    # Load token index (original position)
    idx = tl.load(token_indices_ptr + pid)
    # Load expert output vector for this update as float32
    # We will loop over H in chunks of BLOCK_H and do one atomic add per chunk
    # This keeps atomics minimal: one per chunk per update.
    h = 0
    while h < H:
        offs = h + tl.arange(0, BLOCK_H)
        mask = offs < H
        v = tl.load(expert_outputs_ptr + pid * H + offs, mask=mask, other=0.0)
        # Atomic add into the output row
        out_row_ptr = out_ptr + idx * H + offs
        tl.atomic_add(out_row_ptr, v, mask=mask)
        h += BLOCK_H


def _choose_block_h_and_warps(H: int):
    # Heuristics for chunk size and num_warps based on H
    if H >= 1024:
        return 1024, 8
    elif H >= 512:
        return 512, 4
    else:
        return 256, 2


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add:
            output[token_indices[i]] += expert_outputs[i, :]
        Accumulates in float32 (Triton atomic_add supports fp32), returns bfloat16 like the original.
        """
        # Ensure device and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Inputs must be on CUDA"
        H = final_hidden_states.shape[1]
        N = token_indices.shape[0]

        # Clone input and cast to fp32 for accumulation
        out_fp32 = final_hidden_states.clone().to(torch.float32)
        expert_outputs_fp32 = expert_outputs.to(torch.float32)
        token_indices_i32 = token_indices.to(torch.int32)

        # Grid: one program per update
        grid = (N,)

        # Choose BLOCK_H and num_warps
        BLOCK_H, num_warps = _choose_block_h_and_warps(H)

        # Launch Triton kernel
        scatter_add_rows_kernel[grid](
            out_fp32,
            token_indices_i32,
            expert_outputs_fp32,
            N=N,
            H=H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=2,
        )

        # Cast back to bfloat16 to match original API
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
