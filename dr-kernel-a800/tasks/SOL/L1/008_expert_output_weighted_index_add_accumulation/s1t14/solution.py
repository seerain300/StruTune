import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,     shape [N]
    M: tl.constexpr,  # total rows in output
    N: tl.constexpr,  # number of source rows to add
    H: tl.constexpr,  # hidden dimension
    BLOCK_H: tl.constexpr,  # tile size over hidden dim
):
    # One program per source row
    row = tl.program_id(0)
    if row >= N:
        return

    # Load the token index for this source row
    idx = tl.load(indices_ptr + row)
    # Clamp index to [0, M), even though token_indices is generated in [0, M).
    # Triton allows negative indices, but ensure non-negative to be safe.
    idx = tl.maximum(idx, 0)

    # Iterate over the hidden dimension in tiles
    j = 0
    while j < H:
        offsets = j + tl.arange(0, BLOCK_H)
        mask = offsets < H

        # Compute pointers
        src_row_ptr = src_ptr + row * H + offsets
        out_row_ptr = out_ptr + idx * H + offsets

        # Load source vector (masked), default 0 for out-of-bounds
        val = tl.load(src_row_ptr, mask=mask, other=0.0)

        # Atomic add into the output row
        tl.atomic_add(out_row_ptr, val, mask=mask)

        j += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(
        self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor
    ):
        """
        final_hidden_states: (M, H), bfloat16, device
        expert_outputs: (N, H), bfloat16, device
        token_indices: (N,), int64 or int32, device
        Returns: output (M, H), bfloat16, device, with scatter-add semantics.
        """
        # Ensure contiguity and dtypes
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA"
        M, H = final_hidden_states.shape
        N = expert_outputs.shape[0]

        # Clone to preserve original buffer
        output = final_hidden_states.clone()

        # Triton prefers int32 for indices
        if token_indices.dtype != torch.int32:
            indices32 = token_indices.to(torch.int32)
        else:
            indices32 = token_indices

        # Ensure src and out are contiguous
        src = expert_outputs.contiguous()
        out = output.contiguous()

        # Choose BLOCK_H: use 512 if H <= 512, else 256
        BLOCK_H = 512 if H <= 512 else 256

        # Launch one program per source row
        grid = (N,)

        scatter_add_rows_kernel[grid](
            out, src, indices32,
            M, N, H,
            BLOCK_H=BLOCK_H,
            num_warps=8,   # more warps for better throughput
            num_stages=1,  # simple, memory-bound kernel
        )

        return out


def run(*args):
    return ModelNew()(*args)
