import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,    shape [N]
    M: tl.constexpr,  # total rows in out_ptr/src_ptr
    H: tl.constexpr,  # hidden size
    BLOCK_H: tl.constexpr,  # tile size along hidden dimension
):
    # One Triton program handles one source row i
    row_i = tl.program_id(axis=0)
    if row_i >= tl.num_programs(axis=0):
        return

    # Load token index for this row (destination row in output)
    tok = tl.load(indices_ptr + row_i)
    # Compute base offsets for out and src rows
    out_row_base = tok * H
    src_row_base = row_i * H

    # Vectorized tile loop over hidden dimension
    for h in range(0, H, BLOCK_H):
        cols = h + tl.arange(0, BLOCK_H)
        mask = cols < H

        # Load source vector for this row and tile
        src_ptrs = src_ptr + src_row_base + cols
        vals = tl.load(src_ptrs, mask=mask, other=0.0)

        # Atomic add into output destination row at the corresponding columns
        out_ptrs = out_ptr + out_row_base + cols
        tl.atomic_add(out_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add along dim=0:
        output = final_hidden_states.clone()
        output[token_indices[i]] += expert_outputs[i] for i in [0, N)
        """
        # Ensure inputs are CUDA tensors for Triton
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Inputs must be on CUDA for Triton."

        M = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]
        assert expert_outputs.shape == (N, H), "expert_outputs must have shape [N, H]"
        assert token_indices.shape == (N,), "token_indices must have shape [N]"

        # Heuristic: for very small N, PyTorch's index_add may be faster than launching a custom kernel
        if N < 2048:
            output = torch.zeros((M, H), dtype=final_hidden_states.dtype, device=final_hidden_states.device)
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return output

        # Prepare output and ensure contiguity
        output = final_hidden_states.contiguous()

        # Ensure contiguous layout for sources and proper dtypes
        src = expert_outputs.contiguous()
        indices_i32 = token_indices.to(torch.int32)

        # Choose kernel configuration that performed best: BLOCK_H=256, num_warps=4, num_stages=2
        BLOCK_H = 256
        grid = (N,)

        scatter_add_rows_kernel[grid](
            output, src, indices_i32,
            M, H,
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=2,
        )
        return output


def run(*args):
    return ModelNew()(*args)
