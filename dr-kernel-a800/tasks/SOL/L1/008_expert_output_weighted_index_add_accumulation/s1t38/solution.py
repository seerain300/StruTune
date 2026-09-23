import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *bfloat16, shape [M, H]
    src_ptr,          # *bfloat16, shape [N, H]
    indices_ptr,      # *int32,    shape [N]
    M: tl.constexpr,  # rows in output
    H: tl.constexpr,  # hidden size
    N: tl.constexpr,  # number of source rows
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # One program per source row i
    i = tl.program_id(0)

    # Guard against out-of-range programs (in case grid > N)
    if i >= N:
        return

    # Load the target row index for this source row
    idx = tl.load(indices_ptr + i)

    # Base offsets for output and source rows
    # out[i_row, :]: i_row = idx
    # src[i, :]: i is the program id
    # We will process H in tiles of BLOCK_H, masked for tails
    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask = h_offsets < H

        # Compute pointers
        out_row_ptr = out_ptr + idx * H + h_offsets
        src_row_ptr = src_ptr + i * H + h_offsets

        # Load values (masked for tail)
        src_vals = tl.load(src_row_ptr, mask=mask, other=0.0)

        # Atomic add into output row
        # out_ptr is non-const (writable), tl.atomic_add expects non-const
        tl.atomic_add(out_row_ptr, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        # Ensure device and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA device"
        device = final_hidden_states.device

        # Dimensions
        M = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]
        assert expert_outputs.shape[1] == H, "expert_outputs hidden size must match final_hidden_states"
        assert token_indices.shape[0] == N, "token_indices must match number of expert outputs"

        # Output: we'll fill via atomic adds; initialize to zeros
        output = torch.empty((M, H), dtype=torch.bfloat16, device=device)
        # Make sure inputs are contiguous
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.to(torch.int32).contiguous()

        # Launch Triton kernel: one program per source row
        grid = (N,)
        # Choose tile size and launch config
        BLOCK_H = 256
        num_warps = 4
        num_stages = 2

        scatter_add_rows_kernel[grid](
            output, expert_outputs, token_indices,
            M, H, N, BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return output


def run(*args):
    return ModelNew()(*args)
