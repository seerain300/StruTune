import torch
import triton
import triton.language as tl


@triton.jit
def _index_add_rows_kernel(
    out_ptr,         # *half (output buffer, clone of final_hidden_states)
    expert_ptr,      # *half (expert_outputs)
    indices_ptr,     # *int32 (token_indices)
    N,               # int32: number of selected tokens
    H,               # int32: hidden size
    BLOCK: tl.constexpr,
):
    # One program per selected token row
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Load destination row index for this selected token
    idx = tl.load(indices_ptr + pid)
    # idx is expected to be in [0, N). We assume token_indices are valid.

    # Base pointers for this row
    base_out = out_ptr + idx * H
    base_exp = expert_ptr + pid * H

    # Iterate over hidden dimension in chunks of BLOCK
    for r in range(0, H, BLOCK):
        cols = r + tl.arange(0, BLOCK)
        mask = cols < H
        # Load expert vector chunk and atomically add to output
        vals = tl.load(base_exp + cols, mask=mask, other=0.0)
        # Atomic add into output row at 'idx'
        tl.atomic_add(base_out + cols, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        # Ensure contiguity and dtypes
        out = final_hidden_states.clone()
        # Triton prefers int32 indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)
        # Ensure all tensors are contiguous
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()

        N = token_indices.numel()
        H = final_hidden_states.shape[1]

        # Choose BLOCK deterministically: next power of two of H, capped at 256
        # This reduces the number of chunks and improves performance.
        if H <= 64:
            BLOCK = 64
        elif H <= 128:
            BLOCK = 128
        else:
            BLOCK = 256

        # Choose num_warps based on BLOCK
        num_warps = 4 if BLOCK <= 128 else 8
        grid = (N,)

        # Launch kernel: one program per selected token
        _index_add_rows_kernel[grid](
            out, expert_outputs, token_indices, N, H,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
