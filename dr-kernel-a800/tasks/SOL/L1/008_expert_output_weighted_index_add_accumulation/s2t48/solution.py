import torch
import triton
import triton.language as tl


@triton.jit
def _index_add_rows_kernel(
    out_ptr,         # *half (bf16) output buffer
    expert_ptr,      # *half (bf16) expert_outputs
    indices_ptr,     # *int32 token_indices
    N,               # int32: number of selected tokens
    H,               # int32: hidden size
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Destination row index for this selected token
    idx = tl.load(indices_ptr + pid)

    # Loop over hidden dimension in chunks of BLOCK
    r = 0
    while r < H:
        cols = r + tl.arange(0, BLOCK)
        col_mask = cols < H

        # Compute linear offsets
        out_offsets = idx * H + cols
        exp_offsets = pid * H + cols

        # Masked load of expert chunk
        exp_vals = tl.load(expert_ptr + exp_offsets, mask=col_mask, other=0.0)

        # Atomically add into output row at idx
        tl.atomic_add(out_ptr + out_offsets, exp_vals, mask=col_mask)

        r += BLOCK


def _choose_block_and_warps(H: int):
    # Choose a BLOCK size and num_warps based on H
    if H <= 64:
        block = 64
        warps = 2
    elif H <= 128:
        block = 128
        warps = 4
    else:
        block = 256
        warps = 8
    return block, warps


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure CUDA tensors for Triton
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA for Triton."

        # Clone output buffer to avoid in-place modifications
        output = final_hidden_states.clone()

        # Ensure contiguous layouts
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Triton expects int32 indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        N = expert_outputs.shape[0]
        H = expert_outputs.shape[1]

        # Launch configuration
        BLOCK, num_warps = _choose_block_and_warps(H)
        grid = (N,)  # one program per selected token row

        # Launch kernel
        _index_add_rows_kernel[grid](
            output,
            expert_outputs,
            token_indices,
            N,
            H,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
