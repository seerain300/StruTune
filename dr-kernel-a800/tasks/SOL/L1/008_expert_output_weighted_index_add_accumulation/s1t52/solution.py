import torch
import triton
import triton.language as tl


def _select_block_h_and_warps(H: int):
    # Choose BLOCK_H as the next power-of-two of H, clamped to [128, 1024].
    if H <= 128:
        block_h = 128
    else:
        block_h = 1 << (H - 1).bit_length()  # next power of two >= H
        block_h = min(max(block_h, 128), 1024)
    # Heuristic for num_warps: more warps for larger tiles
    num_warps = 8 if block_h >= 512 else 4
    num_stages = 2
    return block_h, num_warps, num_stages


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,    shape [N]
    M: tl.constexpr,  # total rows in out_ptr/src_ptr (used for bounds check)
    H: tl.constexpr,  # hidden size
    BLOCK_H: tl.constexpr,  # tile size along H
):
    row = tl.program_id(0)  # one program per source row
    # If grid > M (shouldn't happen with our launch, but keep safe)
    if row >= M:
        return
    # Load index for this row (token position)
    idx = tl.load(indices_ptr + row)  # int32
    # Iterate over hidden dimension in tiles
    for h_start in range(0, H, BLOCK_H):
        offs = h_start + tl.arange(0, BLOCK_H)
        mask = offs < H
        # Load source row slice (bfloat16)
        src_row_ptr = src_ptr + row * H + offs
        s = tl.load(src_row_ptr, mask=mask, other=0.0)
        # Compute output pointer for this token index and atomic add
        out_row_ptr = out_ptr + idx * H + offs
        tl.atomic_add(out_row_ptr, s, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure tensors are CUDA and dtypes are correct
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Use bfloat16 for outputs and sources."
        M = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]
        # Ensure contiguous
        out = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        # Triton expects int32 indices for address arithmetic
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)
        token_indices = token_indices.contiguous()
        # Select launch configuration
        BLOCK_H, num_warps, num_stages = _select_block_h_and_warps(H)
        # Launch kernel: one program per source row (grid = N)
        grid = (N,)
        scatter_add_rows_kernel[grid](
            out, expert_outputs, token_indices,
            M, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out


def run(*args):
    return ModelNew()(*args)
