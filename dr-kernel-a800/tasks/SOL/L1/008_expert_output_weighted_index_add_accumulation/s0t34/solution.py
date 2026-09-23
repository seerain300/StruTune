import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_kernel(
    output_ptr,            # *bf16 or *fp16, shape [M, H], row-major
    expert_ptr,            # *bf16 or *fp16, shape [N, H], row-major
    index_ptr,             # *int32, shape [N]
    M: tl.constexpr,       # number of output rows (M = batch_size * seq_len)
    H: tl.constexpr,       # number of hidden features
    N: tl.constexpr,       # number of source rows
    BLOCK_H: tl.constexpr, # tile size along H
):
    # Program id: one program per source row
    pid = tl.program_id(axis=0)
    # Each program handles one source row 'pid'
    # Ensure pid in range [0, N)
    # (Grid is set to N, so this is just a safety check in case of overlaunch)
    # We'll skip processing for pid >= N using a mask.
    mask_n = pid < N

    # If pid >= N, nothing to do; mask_n will zero out stores.
    # Compute starting row offset for output
    # Load token index for this source row
    # index_ptr is int32; we keep M and H as int32 for addressing
    idx = tl.load(index_ptr + pid, mask=mask_n, other=0)  # int32
    # idx should be in [0, M). We don't need to check bounds; token_indices is generated validly.

    # Iterate over H in chunks of BLOCK_H
    # We use a compile-time arange for each chunk and mask for partial last chunk.
    # This avoids while loops and ensures correct vectorized addressing.
    h_start = 0
    while h_start < H:
        h_offsets = h_start + tl.arange(0, BLOCK_H)            # shape [BLOCK_H]
        h_mask = h_offsets < H                                  # mask for valid H
        # Only proceed if pid is valid (mask_n) and we have any valid H in this chunk
        # Triton evaluates elementwise; we can combine masks as needed
        # Load expert outputs for this row and chunk
        # expert_ptr is row-major: offset = pid * H + h_offsets
        vals = tl.load(expert_ptr + pid * H + h_offsets, mask=h_mask, other=0.0)
        # Compute destination addresses: idx is the destination row in output
        dest = idx * H + h_offsets                               # shape [BLOCK_H]
        # Atomic add into output
        tl.atomic_add(output_ptr + dest, vals, mask=h_mask)
        h_start += BLOCK_H


def _launch_scatter_add(output: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
    """
    Launch the Triton kernel to perform: output[token_indices[i]] += expert_outputs[i]
    output: (M, H), contiguous
    expert_outputs: (N, H), contiguous
    token_indices: (N,), int64 or int32 (converted to int32)
    """
    assert output.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda
    assert output.shape[0] == token_indices.numel(), "M (output rows) must equal number of indices"
    assert expert_outputs.shape[0] == token_indices.numel(), "N (source rows) must equal number of indices"
    assert output.shape[1] == expert_outputs.shape[1], "Hidden size H must match between output and expert_outputs"

    # Ensure contiguity
    output = output.contiguous()
    expert_outputs = expert_outputs.contiguous()
    # Triton prefers int32 for index arithmetic
    if token_indices.dtype != torch.int32:
        token_indices = token_indices.to(torch.int32)
    token_indices = token_indices.contiguous()

    M, H = output.shape[0], output.shape[1]
    N = token_indices.numel()

    # Choose BLOCK_H and warps based on H
    if H >= 1024:
        BLOCK_H = 256
        num_warps = 8
        num_stages = 2
    elif H >= 256:
        BLOCK_H = 128
        num_warps = 4
        num_stages = 2
    else:
        BLOCK_H = 64
        num_warps = 2
        num_stages = 2

    # Grid: one program per source row
    grid = (N,)

    scatter_add_row_kernel[grid](
        output, expert_outputs, token_indices,
        M=M, H=H, N=N, BLOCK_H=BLOCK_H,
        num_warps=num_warps, num_stages=num_stages,
    )
    return output


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # The reference run returns output = final_hidden_states.clone(); output.index_add_(0, token_indices, expert_outputs)
        # We must match this exactly. Clone to avoid in-place modification of the input buffer.
        output = final_hidden_states.clone()
        _launch_scatter_add(output, expert_outputs, token_indices)
        return output


def run(*args):
    return ModelNew()(*args)
