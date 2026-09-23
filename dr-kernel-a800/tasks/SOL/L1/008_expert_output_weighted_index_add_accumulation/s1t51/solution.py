import torch
import triton
import triton.language as tl


def _select_block_h_and_warps(H: int):
    # Choose BLOCK_H as next power-of-two >= H, clamped to [128, 1024]
    if H <= 128:
        block_h = 128
    else:
        block_h = 1 << (H - 1).bit_length()  # next power of two
        block_h = min(max(block_h, 128), 1024)
    # Heuristic for warps: more warps for larger tiles
    if block_h <= 256:
        num_warps = 4
    elif block_h <= 512:
        num_warps = 8
    else:  # 1024
        num_warps = 8
    return block_h, num_warps


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,       # *const bfloat16, shape [M, H]
    src_ptr,       # *const bfloat16, shape [N, H]
    indices_ptr,   # *const int32,    shape [N]
    M: tl.constexpr,   # total rows in output
    H: tl.constexpr,   # hidden size (columns)
    N: tl.constexpr,   # number of source rows to add
    BLOCK_H: tl.constexpr,  # tile size across hidden dim
):
    # One program handles one source row i
    row_i = tl.program_id(0)
    if row_i >= N:
        return

    # Load token index for this row (int32)
    tok = tl.load(indices_ptr + row_i)  # tok in [0, M)

    # Iterate across hidden dimension in tiles
    for col_start in range(0, H, BLOCK_H):
        offs = col_start + tl.arange(0, BLOCK_H)
        mask = offs < H

        # Load source row vector for this tile
        src_vals = tl.load(
            src_ptr + row_i * H + offs,
            mask=mask,
            other=0.0
        )  # bfloat16 vector

        # Atomic add into output at the target row
        tl.atomic_add(
            out_ptr + tok * H + offs,
            src_vals,
            mask=mask
        )


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add equivalent of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        Assumes:
          - final_hidden_states: [M, H], bfloat16, device CUDA
          - expert_outputs: [N, H], bfloat16, device CUDA
          - token_indices: [N], int64 or int32, values in [0, M)
        """
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA device"
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "dtype must be bfloat16"

        # Make inputs contiguous
        out = final_hidden_states.clone()  # clone to match original semantics
        src = expert_outputs.contiguous()
        M = out.shape[0]
        N = src.shape[0]
        # Triton prefers int32 indices for atomic addressing; cast safely
        indices_i32 = token_indices.to(torch.int32).contiguous()

        H = out.shape[1]
        # Launch one program per source row
        grid = (N,)

        # Choose tile size and warps to minimize loop iterations and maximize parallelism
        BLOCK_H, num_warps = _select_block_h_and_warps(H)
        num_stages = 2

        scatter_add_rows_kernel[grid](
            out, src, indices_i32,
            M, H, N,
            BLOCK_H,
            num_warps=num_warps, num_stages=num_stages
        )

        return out


def run(*args):
    return ModelNew()(*args)
