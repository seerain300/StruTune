import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,    shape [N]
    M: tl.constexpr,  # number of rows (batch_seq_len)
    H: tl.constexpr,  # hidden size
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # One program per source row
    pid = tl.program_id(axis=0)  # pid in [0, N)
    # Bounds check: if pid >= N (shouldn't happen if grid=N, but keep for safety)
    if pid >= N:
        return

    # Load target row index
    idx = tl.load(indices_ptr + pid)
    # Guard against out-of-range indices (shouldn't occur if indices in [0, M))
    if (idx < 0) or (idx >= M):
        return

    # Vector of column offsets for a tile
    col_offsets = tl.arange(0, BLOCK_H)

    # Loop over hidden dimension in tiles
    # Using a simple for-loop over range(0, H, BLOCK_H) is okay here; H is constexpr.
    for start in range(0, H, BLOCK_H):
        offs = start + col_offsets
        mask = offs < H

        # Compute pointers for this row
        out_row_ptr = out_ptr + idx * H + offs
        src_row_ptr = src_ptr + pid * H + offs

        # Load values (masked for tail)
        out_vals = tl.load(out_row_ptr, mask=mask, other=0.0)      # bfloat16
        src_vals = tl.load(src_row_ptr, mask=mask, other=0.0)      # bfloat16

        # Atomic add: accumulate expert output into target row
        tl.atomic_add(out_row_ptr, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized forward: performs scatter-add of expert_outputs into final_hidden_states
        along dim=0 according to token_indices, with atomic adds for duplicate indices.
        """
        # Ensure device/dtype/contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Use bfloat16 tensors."
        assert token_indices.dtype == torch.long, "token_indices must be torch.long (int64)."

        M = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]

        # Clone output for correctness
        output = final_hidden_states.clone()

        # Make inputs contiguous and cast indices to int32 for Triton
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        indices_i32 = token_indices.to(torch.int32)

        # Select tile and launch config heuristically based on H
        if H <= 512:
            BLOCK_H = 512
            num_warps = 8
            num_stages = 3
        elif H <= 1024:
            BLOCK_H = 1024
            num_warps = 8
            num_stages = 3
        else:
            BLOCK_H = 256
            num_warps = 4
            num_stages = 2

        # Launch kernel: one program per source row
        grid = (N,)
        scatter_add_rows_kernel[grid](
            output, expert_outputs, indices_i32,
            M, H, BLOCK_H,
            num_warps=num_warps, num_stages=num_stages,
        )
        return output


def run(*args):
    return ModelNew()(*args)
