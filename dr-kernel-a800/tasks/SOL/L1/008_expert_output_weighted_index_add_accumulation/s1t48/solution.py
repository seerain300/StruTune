import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,    shape [N]
    M: tl.constexpr,  # number of rows in output
    N: tl.constexpr,  # number of expert outputs
    H: tl.constexpr,  # hidden size
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # One program per output row
    src_row = tl.program_id(0)
    if src_row >= M:
        return

    # Loop over hidden dimension in tiles
    h_start = 0
    while h_start < H:
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H

        # Iterate over all N contributions; for each, if index == src_row, add the src row to output
        j = 0
        while j < N:
            idx = tl.load(indices_ptr + j)  # int32
            if idx == src_row:
                src_vals = tl.load(src_ptr + j * H + h_offsets, mask=mask_h, other=0.0)
                tl.atomic_add(out_ptr + src_row * H + h_offsets, src_vals, mask=mask_h)
            j += 1

        h_start += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Validate shapes
        assert final_hidden_states.dim() == 2, "final_hidden_states must be 2D [M, H]"
        assert expert_outputs.dim() == 2, "expert_outputs must be 2D [N, H]"
        assert token_indices.dim() == 1, "token_indices must be 1D [N]"

        M, H = final_hidden_states.shape
        N = expert_outputs.shape[0]

        # Prepare output as clone of input, ensure dtype and contiguity
        out = final_hidden_states.clone()
        out = out.to(torch.bfloat16).contiguous()

        # Ensure expert outputs are contiguous and bfloat16
        expert_outputs = expert_outputs.to(torch.bfloat16).contiguous()

        # Ensure indices are int32 for Triton
        token_indices = token_indices.to(torch.int32)

        # Launch Triton kernel: one program per output row
        BLOCK_H = 256  # good balance for typical H up to 1024
        grid = (M,)
        scatter_add_rows_kernel[grid](
            out, expert_outputs, token_indices,
            M, N, H,
            BLOCK_H=BLOCK_H,
            num_warps=4,   # 4 warps with 256 tile is a strong default
            num_stages=2,  # modest pipelining
        )
        return out


def run(*args):
    return ModelNew()(*args)
