import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *bfloat16, shape [M, H]
    src_ptr,          # *bfloat16, shape [N, H]
    indices_ptr,      # *int32,    shape [N]
    M: tl.constexpr,  # total rows in out_ptr/src_ptr
    H: tl.constexpr,  # hidden size
    BLOCK_H: tl.constexpr,  # tile size along H (e.g., 256)
):
    # Each program handles one source row i
    i = tl.program_id(0)

    # Base pointers for this row
    out_row_ptr = out_ptr + i * H
    src_row_ptr = src_ptr + i * H

    # Iterate over H in tiles
    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask = h_offsets < H

        # Load token index for this row
        idx = tl.load(indices_ptr + i, mask=True, other=0)
        # Bounds check: idx must be in [0, M). If out-of-range, skip (defensive; indices are valid by construction)
        # Triton does not support dynamic if-else on scalars; we proceed under the assumption indices are valid.

        # Load source vector tile
        src_vec = tl.load(src_row_ptr + h_offsets, mask=mask, other=0.0)

        # Compute destination pointer vector
        out_vec_ptr = out_row_ptr + h_offsets

        # Atomic add into destination
        tl.atomic_add(out_vec_ptr, src_vec, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add along dim=0:
          output = final_hidden_states.clone()
          for i in range(N): output[token_indices[i]] += expert_outputs[i]
        """
        # Ensure tensors are contiguous and dtypes are appropriate
        assert final_hidden_states.dim() == 2, "final_hidden_states must be 2D [M, H]"
        assert expert_outputs.dim() == 2, "expert_outputs must be 2D [N, H]"
        assert token_indices.dim() == 1, "token_indices must be 1D [N]"

        M, H = final_hidden_states.shape
        N, H_src = expert_outputs.shape
        assert H_src == H, "expert_outputs H must match final_hidden_states H"
        assert token_indices.numel() == N, "token_indices length must equal expert_outputs rows"
        assert token_indices.dtype in (torch.int32, torch.int64), "token_indices must be integer type"

        # Clone the accumulation buffer
        output = final_hidden_states.clone()

        # Ensure inputs are on the same device and contiguous
        device = final_hidden_states.device
        # Convert indices to int32 for Triton (smaller and faster)
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Triton requires pointers; ensure tensors are contiguous
        src = expert_outputs.contiguous()
        out = output.contiguous()  # we’ll write into a contiguous copy

        # Launch configuration
        grid = (N,)
        # Choose a robust tile size and launch params
        BLOCK_H = 256
        num_warps = 4
        num_stages = 2

        scatter_add_rows_kernel[grid](
            out, src, token_indices,
            M, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        # Return result with the same shape as final_hidden_states
        return out


def run(*args):
    return ModelNew()(*args)
