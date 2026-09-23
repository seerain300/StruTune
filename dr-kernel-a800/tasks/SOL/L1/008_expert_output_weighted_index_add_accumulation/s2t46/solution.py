import torch
import triton
import triton.language as tl


@triton.jit
def _index_add_rows_kernel(
    out_ptr,         # *half, output buffer (final_hidden_states clone)
    expert_ptr,      # *half, expert_outputs
    indices_ptr,     # *int32, token_indices
    N,               # int32, number of selected tokens
    H,               # int32, hidden size
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Destination row index for this selected token
    idx = tl.load(indices_ptr + pid)

    # We will iterate over hidden dimension in chunks of BLOCK and, where possible,
    # combine up to two adjacent chunks into a single vector add to reduce atomic adds.

    # Process chunks in pairs where possible
    # For H <= BLOCK: do it in one shot
    # For H > BLOCK: iterate r from 0 to H in steps of 2 * BLOCK, handle remainder
    for r in range(0, H, BLOCK * 2):
        # First chunk
        cols0 = r + tl.arange(0, BLOCK)
        mask0 = cols0 < H
        row_expert0 = tl.load(expert_ptr + pid * H + cols0, mask=mask0, other=0.0)

        # Second chunk (if within bounds)
        cols1 = cols0 + BLOCK
        mask1 = (cols0 + BLOCK) < H
        row_expert1 = tl.load(expert_ptr + pid * H + cols1, mask=mask1, other=0.0)

        # Combine into a single vector for atomic add
        combined = row_expert0
        combined = tl.where(mask1, combined + row_expert1, combined)

        dest_ptrs = out_ptr + idx * H + cols0
        tl.atomic_add(dest_ptrs, combined, mask=mask0)

    # Handle remainder (if H not divisible by 2*BLOCK)
    remainder = H - ((H // (BLOCK * 2)) * (BLOCK * 2))
    if remainder > 0:
        cols = (H - remainder) + tl.arange(0, remainder)
        mask = cols < H
        row_expert = tl.load(expert_ptr + pid * H + cols, mask=mask, other=0.0)
        dest_ptrs = out_ptr + idx * H + cols
        tl.atomic_add(dest_ptrs, row_expert, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        final_hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        token_indices: torch.Tensor,
    ):
        # Prepare output as a clone of the accumulation buffer
        output = final_hidden_states.clone()

        # Ensure contiguity and dtypes
        expert = expert_outputs.contiguous()
        indices = token_indices.contiguous().to(torch.int32)

        # Shapes
        N = expert.shape[0]  # number of selected tokens
        H = expert.shape[1]  # hidden size

        # Use BLOCK=256 and a higher number of warps for throughput
        BLOCK = 256
        num_warps = 8

        # Launch one program per selected token row
        grid = (N,)
        _index_add_rows_kernel[grid](
            output,
            expert,
            indices,
            N,
            H,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
