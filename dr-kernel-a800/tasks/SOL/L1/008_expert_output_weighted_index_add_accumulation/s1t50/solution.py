import torch
import triton
import triton.language as tl


def _select_block_h_and_warps(H: int):
    # Choose BLOCK_H as next power-of-two of H, clamped to [128, 1024].
    if H <= 128:
        block_h = 128
    else:
        # next power of two >= H
        block_h = 1 << (H - 1).bit_length()
        block_h = min(max(block_h, 128), 1024)
    # Heuristic for num_warps based on tile size
    if block_h <= 256:
        num_warps = 4
    elif block_h <= 512:
        num_warps = 8
    else:  # 1024
        num_warps = 8
    num_stages = 2
    return block_h, num_warps, num_stages


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,    shape [N]
    N: tl.constexpr,  # total source rows
    H: tl.constexpr,  # hidden size
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # One program per source row
    row = tl.program_id(0)  # in [0, N)

    # Load token index for this row
    tok = tl.load(indices_ptr + row)  # int32

    # Vector of column offsets for the tile
    offs_h = tl.arange(0, BLOCK_H)
    mask = offs_h < H

    # Compute pointers for this row
    out_row_ptr = out_ptr + tok * H
    src_row_ptr = src_ptr + row * H

    # Load the source vector for this row, masked for tail
    vals = tl.load(src_row_ptr + offs_h, mask=mask, other=0.0)  # bfloat16

    # Atomic add into the output at the corresponding token row
    tl.atomic_add(out_row_ptr + offs_h, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-based scatter-add:
        - output = final_hidden_states.clone()
        - for i in range(N): output[token_indices[i]] += expert_outputs[i]
        """
        # Ensure tensors are on CUDA
        assert final_hidden_states.is_cuda, "final_hidden_states must be on CUDA"
        assert expert_outputs.is_cuda, "expert_outputs must be on CUDA"
        assert token_indices.is_cuda, "token_indices must be on CUDA"

        M = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]

        # Clone output (ensures correct initial state)
        output = final_hidden_states.clone()

        # Ensure contiguity and dtypes
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Triton prefers int32 indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Select kernel config
        BLOCK_H, num_warps, num_stages = _select_block_h_and_warps(H)

        # Launch kernel: one program per source row
        grid = (N,)

        scatter_add_rows_kernel[grid](
            output, expert_outputs, token_indices,
            N, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return output


def run(*args):
    return ModelNew()(*args)
