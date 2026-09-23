import math
import torch
import triton
import triton.language as tl


@triton.jit
def _index_add_rows_kernel(
    out_ptr,        # *half: output buffer (contiguous), we accumulate into it
    expert_ptr,     # *half: expert outputs (N, H), contiguous
    indices_ptr,    # *int32: token indices (N,), contiguous
    N,              # int32: number of selected tokens
    H,              # int32: hidden size
    BLOCK: tl.constexpr,
):
    # One program per selected token row
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Destination row index for this selected token
    idx = tl.load(indices_ptr + pid)
    # idx is expected to be in [0, N). token_indices must be valid.

    # Base offsets for out and expert rows (in elements)
    out_row_base = idx * H
    expert_row_base = pid * H

    # Column offsets [0, 1, ..., BLOCK-1]; BLOCK >= H ensures full coverage for common sizes
    offs = tl.arange(0, BLOCK)

    # Load the hidden vector for this selected token
    vals = tl.load(expert_ptr + expert_row_base + offs)

    # Atomically add into the destination row
    tl.atomic_add(out_ptr + out_row_base + offs, vals)


def _next_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << ((x - 1).bit_length())


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized index_add along dim=0 with accumulation:
        output[token_indices[i]] += expert_outputs[i] for all i.
        - final_hidden_states: (batch_seq_len, hidden_size), bfloat16, CUDA
        - expert_outputs: (num_selected_tokens, hidden_size), bfloat16, CUDA
        - token_indices: (num_selected_tokens,), int32/int64, CUDA
        Returns updated final_hidden_states with expert contributions added.
        """
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All inputs must be on CUDA for Triton."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Expect bfloat16 tensors."
        assert token_indices.dtype in (torch.int32, torch.int64), "token_indices must be int32 or int64."

        # Ensure contiguity
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Clone to match original semantics (accumulate into a new buffer)
        out = final_hidden_states.clone()

        N = expert_outputs.shape[0]
        H = expert_outputs.shape[1]

        # Choose BLOCK as next power-of-two of H, capped at 256, so BLOCK >= H for H up to 1024
        BLOCK = _next_power_of_two(H)
        BLOCK = min(BLOCK, 256)

        # Launch one program per selected token row
        grid = (N,)

        # Choose num_warps based on BLOCK
        if BLOCK <= 64:
            num_warps = 2
        elif BLOCK <= 128:
            num_warps = 4
        else:
            num_warps = 8
        num_stages = 2

        _index_add_rows_kernel[grid](
            out, expert_outputs, token_indices.int32(), N, H, BLOCK=BLOCK,
            num_warps=num_warps, num_stages=num_stages
        )

        return out


def run(*args):
    return ModelNew()(*args)
