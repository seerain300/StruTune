import torch
import triton
import triton.language as tl


@triton.jit
def _index_add_rows_kernel(
    out_ptr,         # *half, output buffer
    expert_ptr,      # *half, expert_outputs
    indices_ptr,     # *int32, token_indices
    N,               # int32, number of selected tokens
    H,               # int32, hidden size
    BLOCK: tl.constexpr,
):
    # One program per selected token row
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Destination row index for this selected token
    idx = tl.load(indices_ptr + pid)

    # Iterate over hidden dimension in chunks of BLOCK
    r = 0
    while r < H:
        cols = r + tl.arange(0, BLOCK)
        mask = cols < H

        # Compute linear offsets
        out_offsets = idx * H + cols
        exp_offsets = pid * H + cols

        # Load chunk from expert_outputs (masked for tail)
        vals = tl.load(expert_ptr + exp_offsets, mask=mask, other=0.0)

        # Atomically add into the output row
        tl.atomic_add(out_ptr + out_offsets, vals, mask=mask)

        r += BLOCK


def _next_power_of_two(n: int, cap: int = 256) -> int:
    # Next power of two >= n, capped at 'cap'
    if n <= 64:
        return 64
    elif n <= 128:
        return 128
    else:
        return min(cap, 256)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure tensors are on CUDA
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA."

        # Clone output buffer to preserve original
        out = final_hidden_states.contiguous().clone()
        expert = expert_outputs.contiguous()
        indices = token_indices.contiguous().to(torch.int32)

        # Dimensions
        N = expert.shape[0]
        H = expert.shape[1]

        # Choose BLOCK and num_warps based on H
        BLOCK = _next_power_of_two(H, cap=256)
        if BLOCK <= 64:
            num_warps = 2
        elif BLOCK <= 128:
            num_warps = 4
        else:
            num_warps = 8

        # Launch one program per selected token row
        grid = (N,)
        _index_add_rows_kernel[grid](out, expert, indices, N, H, BLOCK=BLOCK, num_warps=num_warps, num_stages=2)
        return out


def run(*args):
    return ModelNew()(*args)
