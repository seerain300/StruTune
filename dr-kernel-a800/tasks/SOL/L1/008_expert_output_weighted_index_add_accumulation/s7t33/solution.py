import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_per_row_chunked_kernel(
    out_ptr,        # *bf16, pointer to out tensor [N, H], contiguous
    src_ptr,        # *bf16, pointer to src tensor [M, H], contiguous
    indices_ptr,    # *int32, pointer to indices tensor [M], contiguous
    M,              # int32, number of source rows (runtime)
    N,              # int32, number of rows in out (runtime)  [not used directly, kept for clarity]
    H,              # int32, hidden size (runtime)
    BLOCK_SIZE: tl.constexpr,  # chunk size along H (compile-time constant)
):
    # One program per source row
    pid = tl.program_id(axis=0)
    if pid >= M:
        return

    # Destination row in out
    dst = tl.load(indices_ptr + pid)  # int32

    # Process hidden dimension in chunks of BLOCK_SIZE
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H

        # Compute linear offsets for source and destination
        src_offsets = pid * H + offs
        out_offsets = dst * H + offs

        # Load source values for this chunk (masked)
        src_vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)

        # Atomic add into out: out[dst, offs] += src_vals
        tl.atomic_add(out_ptr + out_offsets, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure tensors are on CUDA
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA."

        # Clone to match original behavior
        out = final_hidden_states.clone()

        # Ensure contiguity for simple pointer arithmetic
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous().to(torch.int32)

        M = expert_outputs.shape[0]
        N = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]

        # Grid: one program per source row
        grid = (M,)

        # Choose a modest BLOCK_SIZE; 128 works well across a range of H.
        BLOCK_SIZE = 128

        scatter_add_per_row_chunked_kernel[grid](
            out, expert_outputs, token_indices,
            M, N, H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=1,  # simple, reliable configuration
            num_stages=1,
        )

        return out


def run(*args):
    return ModelNew()(*args)
