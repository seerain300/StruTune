import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_kernel(
    output_ptr,            # *fp16/bf16 pointer, shape [M, H]
    expert_ptr,            # *fp16/bf16 pointer, shape [N, H]
    index_ptr,             # *int32 pointer, shape [N]
    N: tl.constexpr,       # number of source rows
    H: tl.constexpr,       # hidden size (columns)
    BLOCK_H: tl.constexpr  # chunk size over H
):
    pid = tl.program_id(0)  # one program per source row n
    if pid >= N:
        return

    # Load token index for this source row
    idx = tl.load(index_ptr + pid)
    # If idx is out of range (shouldn't happen with provided data), skip
    if idx < 0:
        return

    # Iterate over H in chunks of BLOCK_H
    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_H)
        mask = offs < H

        # Load the corresponding expert_outputs row chunk
        vals = tl.load(expert_ptr + pid * H + offs, mask=mask, other=0.0)

        # Compute linear destination addresses: out[idx, offs] -> idx * H + offs
        dest = idx * H + offs

        # Atomic add into output
        tl.atomic_add(output_ptr + dest, vals, mask=mask)

        start += BLOCK_H


def _choose_launch_params(H: int):
    # Adaptive tuning for performance
    if H >= 1024:
        return 256, 8, 3
    elif H >= 256:
        return 128, 4, 2
    else:
        return 64, 2, 2


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure device and dtypes
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA"
        assert final_hidden_states.shape == (final_hidden_states.shape[0], final_hidden_states.shape[1]), "final_hidden_states must be 2D (M, H)"
        assert expert_outputs.shape[0] == token_indices.shape[0], "expert_outputs and token_indices lengths must match"
        M, H = final_hidden_states.shape
        N = token_indices.shape[0]

        # Make sure expert_outputs is contiguous row-major for coalesced loads
        expert_outputs = expert_outputs.contiguous()

        # Triton kernel expects int32 indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Allocate output (clone input to match reference behavior)
        output = final_hidden_states.clone()

        # Launch Triton kernel: one program per source row
        BLOCK_H, num_warps, num_stages = _choose_launch_params(H)
        grid = (N,)

        scatter_add_row_kernel[grid](
            output, expert_outputs, token_indices,
            N=N, H=H, BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=num_stages
        )

        return output


def run(*args):
    return ModelNew()(*args)
