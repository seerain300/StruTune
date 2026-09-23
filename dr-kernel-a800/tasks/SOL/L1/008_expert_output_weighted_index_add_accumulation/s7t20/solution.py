import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_per_row_kernel(
    out_ptr,        # *bf16, shape (N, H)
    src_ptr,        # *bf16, shape (M, H)
    indices_ptr,    # *int32, shape (M,)
    N: tl.constexpr,  # number of rows in out (batch_seq_len)
    H: tl.constexpr,  # hidden size
    M: tl.constexpr,  # number of selected tokens (M = N * num_experts_per_tok)
    BLOCK_H: tl.constexpr,  # loop unroll factor over hidden dim (e.g., 32)
):
    pid = tl.program_id(0)  # row index in [0, N)
    # Guard: if pid >= N, return (defensive; grid should be N, but keep safe)
    if pid >= N:
        return

    # Compute destination row index for this selected token
    # indices_ptr[pid] gives the token position this expert output belongs to
    row_idx = tl.load(indices_ptr + pid)
    # Optional guard in case indices are out of bounds (shouldn't happen if generated correctly)
    if row_idx < 0 or row_idx >= N:
        return

    # Loop over hidden dimension in chunks of BLOCK_H for better codegen
    for j in range(0, H, BLOCK_H):
        # Iterate over the small chunk
        for k in range(BLOCK_H):
            col = j + k
            # Scalar load from src row pid at column col
            val = tl.load(src_ptr + pid * H + col)
            # Atomic add into out[row_idx, col]
            tl.atomic_add(out_ptr + row_idx * H + col, val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Match reference: clone first
        out = final_hidden_states.clone()

        # Ensure contiguous and proper dtypes
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        # Triton expects int32 for indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)
        token_indices = token_indices.contiguous()

        N = out.shape[0]  # batch_seq_len
        H = out.shape[1]
        M = expert_outputs.shape[0]  # should equal N * num_experts_per_tok in your setup

        # Launch 1D grid over rows
        grid = (N,)
        # Choose a modest unroll factor for the hidden dimension loop
        BLOCK_H = 32

        scatter_add_per_row_kernel[grid](
            out, expert_outputs, token_indices,
            N, H, M, BLOCK_H,
            num_warps=4,
        )
        return out


def run(*args):
    return ModelNew()(*args)
