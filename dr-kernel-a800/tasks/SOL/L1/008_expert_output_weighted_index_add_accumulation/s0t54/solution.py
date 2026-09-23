import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_kernel(
    output_ptr,            # *bf16 or *fp16, shape [M, H], row-major contiguous
    expert_ptr,            # *bf16 or *fp16, shape [N, H], row-major contiguous
    index_ptr,             # *int32, shape [N]
    M: tl.int32,           # number of rows in output (M = batch_size * seq_len)
    H: tl.int32,           # hidden size
    BLOCK_H: tl.constexpr, # tile size along H
):
    pid = tl.program_id(axis=0)  # program id along N (one per source row)
    # If grid > N, early return (safety if grid is larger than N)
    if pid >= tl.num_programs(axis=0):
        return

    # Load token index for this source row
    dest_row = tl.load(index_ptr + pid)  # int32

    # Base pointer for the current source row's expert data
    base_expert = expert_ptr + pid * H

    # Iterate over H in chunks
    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_H)
        mask = offs < H

        # Load H-chunk from expert output row
        expert_vals = tl.load(base_expert + offs, mask=mask, other=0.0)

        # Compute output addresses: output[dest_row, offs]
        out_ptrs = output_ptr + dest_row * H + offs

        # Atomic add to output
        tl.atomic_add(out_ptrs, expert_vals, mask=mask)

        start += BLOCK_H


def _choose_kernel_config(H: int):
    # Choose BLOCK_H, num_warps, num_stages based on H
    if H >= 8192:
        BLOCK_H = 512
        num_warps = 8
        num_stages = 4
    elif H >= 2048:
        BLOCK_H = 256
        num_warps = 8
        num_stages = 4
    elif H >= 512:
        BLOCK_H = 128
        num_warps = 4
        num_stages = 3
    else:
        BLOCK_H = 64
        num_warps = 2
        num_stages = 2
    return BLOCK_H, num_warps, num_stages


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We perform the atomic accumulation directly in a Triton kernel.
        """
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA for Triton kernel"
        assert final_hidden_states.dim() == 2, "final_hidden_states must be 2D [M, H]"
        assert expert_outputs.dim() == 2, "expert_outputs must be 2D [N, H]"
        assert token_indices.dim() == 1, "token_indices must be 1D [N]"
        M, H_out = final_hidden_states.shape
        N, H_exp = expert_outputs.shape
        assert H_out == H_exp, "hidden_size must match between output and expert_outputs"
        assert token_indices.dtype in (torch.int32, torch.int64), "token_indices must be int32 or int64"

        # Ensure dtypes and contiguity
        # Make sure token_indices are int32 for efficient address arithmetic
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        output = final_hidden_states  # accumulate in-place as per original semantics

        # Grid: one program per source row (N)
        BLOCK_H, num_warps, num_stages = _choose_kernel_config(H_out)
        grid = (N,)  # one program per row

        scatter_add_row_kernel[grid](
            output, expert_outputs, token_indices,
            M, H_out,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return output


def run(*args):
    return ModelNew()(*args)
