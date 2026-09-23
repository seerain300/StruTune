import math
import torch
import triton
import triton.language as tl


@triton.jit
def _index_add_rows_kernel(
    out_ptr,         # *half (final_hidden_states clone), atomically added into
    expert_ptr,      # *half (expert_outputs)
    indices_ptr,     # *int32 (token_indices)
    N,               # int32: number of selected tokens
    H,               # int32: hidden size
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Destination row index for this selected token
    idx = tl.load(indices_ptr + pid)

    # Iterate over hidden dimension in chunks of BLOCK
    col_start = 0
    while col_start < H:
        cols = col_start + tl.arange(0, BLOCK)
        mask = cols < H

        # Load chunk from expert_outputs[pid, cols]
        expert_vals = tl.load(
            expert_ptr + pid * H + cols,
            mask=mask,
            other=0.0,
        )

        # Compute destination pointers: out[idx, cols]
        out_ptrs = out_ptr + idx * H + cols
        # Atomic add: out[idx, cols] += expert_vals
        tl.atomic_add(out_ptrs, expert_vals, mask=mask)

        col_start += BLOCK


def _choose_block_and_warps(H: int):
    # Deterministic selection to reduce lanes for small H and maintain throughput for larger H
    if H <= 64:
        return 64, 2
    elif H <= 128:
        return 128, 4
    else:
        return 256, 8


class ModelNew(torch.nn.Module):
    def forward(
        self,
        final_hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        # Ensure tensors are on CUDA
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA device"

        # Clone to preserve original semantics
        out = final_hidden_states.clone()
        out = out.contiguous()

        # Ensure inputs are contiguous and dtypes are appropriate
        expert_outputs = expert_outputs.contiguous()
        # Triton prefers int32 for indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)
        token_indices = token_indices.contiguous()

        N = expert_outputs.shape[0]
        H = expert_outputs.shape[1]

        # Launch one program per selected token row
        BLOCK, num_warps = _choose_block_and_warps(H)
        grid = (N,)
        num_stages = 2

        _index_add_rows_kernel[grid](
            out,
            expert_outputs,
            token_indices,
            N,
            H,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out


def run(*args):
    return ModelNew()(*args)
