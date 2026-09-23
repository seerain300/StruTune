import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_atomic_kernel(
    output_ptr,          # *output* (M, H), contiguous
    expert_outputs_ptr,  # *expert_outputs* (N, H), contiguous
    token_indices_ptr,   # *token_indices* (N,), int64
    N, H,                # int32
    BLOCK_H: tl.constexpr,
):
    # Each program handles one source row i along axis 0, and a block of H along axis 1
    i = tl.program_id(0)
    h_block = tl.program_id(1)

    # Guard (in case grid > N): although we set grid=(N, ceil_div(H, BLOCK_H)), keep safety
    if i >= N:
        return

    # Load target row index for this source row i
    # token_indices_ptr is int64 on device; Triton will treat it as 64-bit integer
    idx = tl.load(token_indices_ptr + i)  # scalar int64

    # Compute H offsets for this tile
    h_offsets = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = h_offsets < H

    # Compute source and destination pointers
    # expert_outputs is [N, H] contiguous: row i, then H features
    src_ptrs = expert_outputs_ptr + i * H + h_offsets
    # output is [M, H] contiguous: row idx, then H features
    dst_ptrs = output_ptr + idx * H + h_offsets

    # Load source values (bf16), add to destination (bf16) using atomic
    vals = tl.load(src_ptrs, mask=mask, other=0.0)
    # Atomic add to output
    tl.atomic_add(dst_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of scatter-add along dim=0:
        output[token_indices[i]] += expert_outputs[i] for all i.
        We ensure each source row i is processed exactly once by setting BLOCK_N=1 in the grid.
        """
        # Clone to initialize output (matches reference behavior)
        output = final_hidden_states.clone()

        # Ensure tensors are on CUDA
        assert output.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA."

        # Shapes
        M, H = output.shape
        N = expert_outputs.shape[0]
        # Hints for contiguity
        assert final_hidden_states.is_contiguous(), "final_hidden_states must be contiguous"
        assert expert_outputs.is_contiguous(), "expert_outputs must be contiguous"
        # token_indices can be non-contiguous; we don't require it, but ensure dtype is long
        assert token_indices.dtype in (torch.int64, torch.int32), "token_indices must be integer type"

        # Choose BLOCK_H based on H for performance
        if H >= 1024:
            BLOCK_H = 256
            num_warps = 8
            num_stages = 2
        else:
            BLOCK_H = 128
            num_warps = 4
            num_stages = 2

        # Grid: axis 0 = N, axis 1 = ceil_div(H, BLOCK_H)
        grid = (N, triton.cdiv(H, BLOCK_H))

        scatter_add_atomic_kernel[grid](
            output, expert_outputs, token_indices,
            N, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=num_stages,
        )

        return output


def run(*args):
    return ModelNew()(*args)
