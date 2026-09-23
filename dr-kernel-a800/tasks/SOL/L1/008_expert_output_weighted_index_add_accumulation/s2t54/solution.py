import torch
import triton
import triton.language as tl


@triton.jit
def _index_add_rows_kernel(
    out_ptr,         # *half (final_hidden_states clone), we will atomically add into it
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
    cols = tl.arange(0, BLOCK)
    num_chunks = (H + BLOCK - 1) // BLOCK
    for chunk in range(0, num_chunks):
        col_start = chunk * BLOCK
        col_offsets = col_start + cols
        mask = col_offsets < H

        # Load the hidden vector chunk for this selected token
        in_ptrs = expert_ptr + pid * H + col_offsets
        val = tl.load(in_ptrs, mask=mask, other=0.0)

        # Compute output pointers for this destination row
        out_ptrs = out_ptr + idx * H + col_offsets

        # Atomically add into the output row
        # Using float32 for atomic add (Triton supports atomic_add on fp32)
        # Convert bfloat16 to fp32 for the atomic, then cast back if needed
        val_fp32 = val.to(tl.float32)
        tl.atomic_add(out_ptrs, val_fp32, mask=mask)


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure contiguity and dtype/device
        out = final_hidden_states.clone()
        # Triton prefers int32 for indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)
        if not out.is_contiguous():
            out = out.contiguous()
        if not expert_outputs.is_contiguous():
            expert_outputs = expert_outputs.contiguous()
        if not token_indices.is_contiguous():
            token_indices = token_indices.contiguous()

        # Launch configuration: one program per row; BLOCK=256; num_warps=4, num_stages=2
        N = expert_outputs.shape[0]
        H = expert_outputs.shape[1]

        grid = (N,)
        # Using BLOCK=256 gives good throughput and minimal loop iterations for common H<=1024
        kernel = _index_add_rows_kernel
        kernel[grid](
            out, expert_outputs, token_indices,
            N, H,
            BLOCK=256,
            num_warps=4,  # restore to previously faster setting
            num_stages=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
