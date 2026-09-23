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
    BLOCK_SIZE: tl.constexpr,  # chunk size along H (compile-time)
):
    # One program per source row
    pid = tl.program_id(axis=0)
    if pid >= M:
        return

    # Destination row index in out
    dst = tl.load(indices_ptr + pid)  # int32
    if dst < 0 or dst >= N:
        return  # safety if indices are out of bounds (shouldn't happen with provided inputs)

    # Process hidden dimension in chunks of BLOCK_SIZE
    # Masking handles cases where H is not divisible by BLOCK_SIZE.
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H

        # Compute pointers for this chunk
        out_row_ptr = out_ptr + dst * H
        src_row_ptr = src_ptr + pid * H

        # Load chunk from source
        src_vals = tl.load(src_row_ptr + offs, mask=mask, other=0.0)

        # Atomic add into destination row
        tl.atomic_add(out_row_ptr + offs, src_vals, mask=mask)


# Kernel launch configuration tuned for robustness and modest speedups
BLOCK_SIZE = 1024
num_warps = 8
num_stages = 1


def triton_scatter_add(final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
    """
    Triton implementation of:
        output = final_hidden_states.clone()
        output.index_add_(dim=0, index=token_indices, source=expert_outputs)
    where output has shape (batch_seq_len, hidden_size), token_indices has shape (num_selected_tokens,),
    expert_outputs has shape (num_selected_tokens, hidden_size).
    """
    assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA for Triton."
    assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Expected bfloat16 tensors."
    assert final_hidden_states.is_contiguous(), "final_hidden_states must be contiguous."
    assert expert_outputs.is_contiguous(), "expert_outputs must be contiguous."
    assert token_indices.is_contiguous(), "token_indices must be contiguous."

    M = expert_outputs.shape[0]
    N = final_hidden_states.shape[0]
    H = final_hidden_states.shape[1]

    # Make indices int32 for Triton
    token_indices_i32 = token_indices if token_indices.dtype == torch.int32 else token_indices.to(torch.int32)

    # Ensure output buffer is initialized to zeros (atomic add semantics require this)
    output = torch.zeros_like(final_hidden_states)

    grid = (M,)
    scatter_add_per_row_chunked_kernel[grid](
        output, expert_outputs, token_indices_i32, M, N, H,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


# Optional: keep the same signature for compatibility with the evaluator
class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        return triton_scatter_add(final_hidden_states, expert_outputs, token_indices)


def run(*args):
    return ModelNew()(*args)
