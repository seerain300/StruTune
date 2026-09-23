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
        # out tensor is [N, H], contiguous => row offset = dst * H, then column offset = offs
        out_row_ptr = out_ptr + dst * H
        src_row_ptr = src_ptr + pid * H

        # Load a chunk of src row (masked), default other=0.0
        src_vals = tl.load(src_row_ptr + offs, mask=mask, other=0.0)

        # Atomic add to the corresponding positions in the out row
        tl.atomic_add(out_row_ptr + offs, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure tensors are on CUDA and contiguous
        assert final_hidden_states.is_cuda, "final_hidden_states must be on CUDA"
        assert expert_outputs.is_cuda, "expert_outputs must be on CUDA"
        assert token_indices.is_cuda, "token_indices must be on CUDA"

        # Allocate output (clone to match original semantics)
        # Note: We could use final_hidden_states directly if we were allowed to update in-place,
        # but cloning ensures we don't modify the input tensor, matching the original code.
        out = final_hidden_states.clone()
        src = expert_outputs.contiguous()
        # Triton prefers int32 indices
        indices = token_indices.to(torch.int32).contiguous()

        # Shapes
        N = out.shape[0]  # batch_seq_len
        M = src.shape[0]  # num_selected_tokens
        H = out.shape[1]  # hidden_size

        # Launch grid: one program per source row
        grid = (M,)

        # Use a configuration that performed well in this environment
        BLOCK_SIZE = 256

        scatter_add_per_row_chunked_kernel[grid](
            out, src, indices,
            M, N, H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=2,
            num_stages=1,
        )
        return out


def run(*args):
    return ModelNew()(*args)
