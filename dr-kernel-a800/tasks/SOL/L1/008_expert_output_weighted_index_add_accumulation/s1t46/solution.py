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
    BLOCK_H: tl.constexpr,  # tile size along H
):
    row_id = tl.program_id(0)  # each program handles one source row
    # Load destination index (token position)
    dest = tl.load(indices_ptr + row_id)  # int32
    # Iterate over hidden dimension in tiles
    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask = h_offsets < H

        # Load source values for this row and tile
        src_row_ptr = src_ptr + row_id * H
        src_vals = tl.load(src_row_ptr + h_offsets, mask=mask, other=0.0)  # bfloat16

        # Compute destination pointers for this tile and atomic add
        out_row_ptr = out_ptr + dest * H
        tl.atomic_add(out_row_ptr + h_offsets, src_vals, mask=mask)


def _select_block_h_and_warps(H: int):
    # Heuristic: for H <= 128, use 128 tiles; else use 256 tiles.
    # Warps: 4 is best in our evaluations across varied H.
    if H <= 128:
        return 128, 4
    else:
        return 256, 4


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-based scatter-add along dim=0:
        output[token_indices[i]] += expert_outputs[i] for i in [0, N).
        """
        # Ensure device and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA"
        out = final_hidden_states.clone()
        out = out.contiguous()
        src = expert_outputs.contiguous()
        indices = token_indices.contiguous().to(torch.int32)

        M = out.shape[0]
        H = out.shape[1]
        N = src.shape[0]

        # Select tile size and warps
        BLOCK_H, num_warps = _select_block_h_and_warps(H)

        # Launch one program per source row
        grid = (N,)
        scatter_add_rows_kernel[grid](
            out, src, indices,
            M, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
