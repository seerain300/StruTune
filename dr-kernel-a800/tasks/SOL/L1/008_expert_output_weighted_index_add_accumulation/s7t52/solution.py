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
    # One Triton program per source row
    pid = tl.program_id(axis=0)
    if pid >= M:
        return

    # Destination row index in out (indices are int32)
    dst = tl.load(indices_ptr + pid)
    if dst < 0 or dst >= N:
        return  # safety if indices are out of bounds (shouldn't happen with provided inputs)

    # Process hidden dimension in chunks of BLOCK_SIZE; mask handles tail.
    # Use row pointers for contiguous [*, H] tensors.
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        # Row pointers: out row at dst, src row at pid
        out_row_ptr = out_ptr + dst * H
        src_row_ptr = src_ptr + pid * H
        # Load a chunk of the source row (masked), add atomically to destination row
        vals = tl.load(src_row_ptr + offs, mask=mask, other=0.0)
        tl.atomic_add(out_row_ptr + offs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized atomic scatter-add:
          final_hidden_states[pid] += expert_outputs[i] for each i with token_indices[i] = pid
        Assumes:
          - final_hidden_states: [N, H], bfloat16, CUDA, contiguous
          - expert_outputs: [M, H], bfloat16, CUDA, contiguous
          - token_indices: [M], int64 (PyTorch default), CUDA
        """
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Inputs must be on CUDA device."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Dtype must be bfloat16."
        assert final_hidden_states.is_contiguous() and expert_outputs.is_contiguous(), "Tensors must be contiguous."

        # Ensure indices are int32 for Triton
        indices_i32 = token_indices.to(torch.int32)

        N = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        M = expert_outputs.shape[0]

        # Output buffer (accumulate in-place semantics as in the original)
        out = final_hidden_states  # write in-place into the provided tensor

        # Launch grid: one program per source row
        grid = (M,)
        # Use BLOCK_SIZE=128 and num_warps=2 which previously delivered strong speedups
        BLOCK_SIZE = 128
        scatter_add_per_row_chunked_kernel[grid](
            out, expert_outputs, indices_i32, M, N, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=2, num_stages=1
        )
        return out


def run(*args):
    return ModelNew()(*args)
