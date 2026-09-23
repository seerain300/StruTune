import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,    shape [N]
    M: tl.constexpr,  # number of rows in output (not used but kept for clarity)
    H: tl.constexpr,  # hidden size
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # Each program handles one source row (i in [0, N))
    row = tl.program_id(0)
    if row >= N:
        return

    # Load the target token index for this row
    tok = tl.load(indices_ptr + row)  # int32

    # Iterate over H in tiles of size BLOCK_H
    for h_start in range(0, H, BLOCK_H):
        offs = h_start + tl.arange(0, BLOCK_H)  # vector of H offsets
        mask = offs < H

        # Load expert outputs for this row at these H offsets
        src_vals = tl.load(src_ptr + row * H + offs, mask=mask, other=0.0)  # bfloat16

        # Atomic add into the output at row tok
        out_add_ptr = out_ptr + tok * H + offs
        tl.atomic_add(out_add_ptr, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Validate dtypes and contiguity
        assert final_hidden_states.dtype == torch.bfloat16, "final_hidden_states must be bfloat16"
        assert expert_outputs.dtype == torch.bfloat16, "expert_outputs must be bfloat16"
        assert token_indices.dtype == torch.int32, "token_indices must be int32"

        # Ensure tensors are contiguous
        out = final_hidden_states.clone()
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        M = out.shape[0]
        N = expert_outputs.shape[0]
        H = out.shape[1]
        assert expert_outputs.shape[1] == H, "expert_outputs second dimension must match hidden size"

        # Launch Triton kernel: one program per source row
        BLOCK_H = 256  # fixed tile size for robust performance
        grid = (N,)
        scatter_add_rows_kernel[grid](
            out, expert_outputs, token_indices,
            M, H, BLOCK_H,
            num_warps=4,   # good balance for BLOCK_H=256
            num_stages=2,  # pipeline loads/stores
        )
        return out


def run(*args):
    return ModelNew()(*args)
