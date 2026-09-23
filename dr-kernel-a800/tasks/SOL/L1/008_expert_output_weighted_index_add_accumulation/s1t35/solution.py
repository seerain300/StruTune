import torch
import triton
import triton.language as tl


def _select_block_h_and_warps(H: int):
    # Choose BLOCK_H as next power-of-two tile size, clamped to [128, 1024].
    # Heuristic: larger tiles for larger H to minimize loop iterations.
    if H <= 128:
        block_h = 128
        num_warps = 4
    elif H <= 256:
        block_h = 256
        num_warps = 4
    elif H <= 512:
        block_h = 512
        num_warps = 8
    else:
        block_h = 1024  # for H >= 1024, use 1024 tiles (often 1 iteration when H==1024)
        num_warps = 8
    return block_h, num_warps


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,    shape [N]
    M: tl.constexpr,  # total rows in out_ptr/src_ptr (bounds check)
    N: tl.constexpr,  # total source rows (launch grid size)
    H: tl.constexpr,  # hidden size
    BLOCK_H: tl.constexpr,  # tile size for H
):
    # One Triton program per source row i
    i = tl.program_id(0)  # i in [0, N)
    if i >= N:
        return

    # Load the token index for this row
    idx = tl.load(indices_ptr + i)  # int32
    if (idx < 0) or (idx >= M):
        return

    # Vectorized traversal over H in tiles of size BLOCK_H
    offs_h = tl.arange(0, BLOCK_H)  # column offsets

    for start in range(0, H, BLOCK_H):
        cols = start + offs_h
        mask = cols < H

        # Load src[i, cols]
        src_row_ptr = src_ptr + i * H + cols
        src_vals = tl.load(src_row_ptr, mask=mask, other=0.0)  # bfloat16

        # Compute output row base pointer
        out_row_ptr = out_ptr + idx * H + cols

        # Atomic add into output
        tl.atomic_add(out_row_ptr, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure device and dtype
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA"
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Use bfloat16 for outputs and sources"
        assert token_indices.dtype == torch.int32, "token_indices must be int32 on device"

        # Clone the initial buffer (no torch indexing/scatter beyond this point)
        output = final_hidden_states.clone()

        M = output.shape[0]
        H = output.shape[1]
        N = expert_outputs.shape[0]

        # Select tile size and warps heuristically based on H
        BLOCK_H, num_warps = _select_block_h_and_warps(H)

        # Launch kernel: one program per source row
        grid = (N,)
        scatter_add_rows_kernel[grid](
            output,                   # out_ptr
            expert_outputs,           # src_ptr
            token_indices,            # indices_ptr
            M=M, N=N, H=H,            # meta-parameters
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=2,
        )
        return output


def run(*args):
    return ModelNew()(*args)
