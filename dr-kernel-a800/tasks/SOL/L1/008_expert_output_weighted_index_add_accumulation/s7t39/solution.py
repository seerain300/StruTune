import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_per_row_chunked_kernel(
    out_ptr,        # *bf16, pointer to out tensor [N, H], contiguous
    src_ptr,        # *bf16, pointer to src tensor [M, H], contiguous
    indices_ptr,    # *int32, pointer to indices tensor [M], contiguous
    M,              # int32, number of source rows
    N,              # int32, number of rows in out (batch_seq_len)
    H,              # int32, hidden size
    BLOCK_SIZE: tl.constexpr,  # chunk size along H (compile-time constant)
):
    # One program per source row
    pid = tl.program_id(axis=0)
    if pid >= M:
        return

    # Destination row in out
    dst = tl.load(indices_ptr + pid)  # int32
    # Guard in case dst is out of bounds (shouldn't happen if indices valid)
    if dst < 0 or dst >= N:
        return

    # Process hidden dimension in chunks of BLOCK_SIZE
    # Masking handles cases where H is not divisible by BLOCK_SIZE.
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H

        # Load source chunk (row pid)
        src_ptrs = src_ptr + pid * H + offs
        src_vals = tl.load(src_ptrs, mask=mask, other=0.0)  # *bf16 vector

        # Compute destination pointers for atomics
        out_ptrs = out_ptr + dst * H + offs

        # Atomically accumulate
        tl.atomic_add(out_ptrs, src_vals, mask=mask)


def triton_scatter_add(final_hidden_states: torch.Tensor,
                       expert_outputs: torch.Tensor,
                       token_indices: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of atomic scatter-add:
      output[i] += expert_outputs[j] for all j where token_indices[j] == i.
    """
    # Ensure tensors are on CUDA and contiguous
    assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
        "All tensors must be on CUDA for Triton execution."
    out = final_hidden_states.contiguous()
    src = expert_outputs.contiguous()
    # Triton expects int32 indices for pointer arithmetic
    indices = token_indices.to(torch.int32).contiguous()

    M = src.shape[0]
    N = out.shape[0]
    H = out.shape[1]

    # Launch 1D grid: one program per source row
    grid = (M,)
    # Use a moderate BLOCK_SIZE to balance loop iterations and performance
    BLOCK_SIZE = 128
    scatter_add_per_row_chunked_kernel[grid](
        out, src, indices,
        M, N, H,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=2,  # simple parallelism per program
        num_stages=1,
    )
    return out


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        # Triton-only computation, no PyTorch index_add
        return triton_scatter_add(final_hidden_states, expert_outputs, token_indices)


def run(*args):
    return ModelNew()(*args)
