import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_atomic_rows_kernel(
    out_ptr,        # *bf16, shape (N, H), contiguous
    src_ptr,        # *bf16, shape (M, H), contiguous
    indices_ptr,    # *int32, shape (M,), contiguous
    N: tl.int32,    # number of rows in out (batch_seq_len)
    M: tl.int32,    # number of expert outputs
    H: tl.int32,    # hidden size
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    # Load token index for this expert output row
    row_idx = tl.load(indices_ptr + pid)  # int32
    # Atomic add across hidden dimension
    for j in range(0, H):
        out_offset = row_idx * H + j
        src_offset = pid * H + j
        val = tl.load(src_ptr + src_offset)
        tl.atomic_add(out_ptr + out_offset, val)


@triton.jit
def copy_rows_kernel(
    out_ptr,        # *bf16, shape (N, H), contiguous
    src_ptr,        # *bf16, shape (M, H), contiguous
    indices_ptr,    # *int32, shape (M,), contiguous
    M: tl.int32,    # number of rows to copy
    H: tl.int32,    # hidden size
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_idx = tl.load(indices_ptr + pid)  # int32
    # Copy src[pid, :] into out[row_idx, :]
    for j in range(0, H):
        out_offset = row_idx * H + j
        src_offset = pid * H + j
        val = tl.load(src_ptr + src_offset)
        tl.store(out_ptr + out_offset, val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton scatter-add implementation:
        out = final_hidden_states.clone()
        out[token_indices[i]] += expert_outputs[i] for all i
        Uses a non-atomic copy kernel when token_indices are unique; otherwise atomic kernel.
        """
        # Ensure inputs are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA."

        # Clone to match reference behavior (index_add on a fresh buffer)
        out = final_hidden_states.clone()
        out = out.contiguous()

        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Cast indices to int32 for Triton pointer arithmetic
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        N = out.shape[0]  # batch_seq_len
        M = expert_outputs.shape[0]
        H = expert_outputs.shape[1]  # hidden size

        # Detect uniqueness of token_indices to choose non-atomic or atomic kernel
        # unique returns (values, inverse); we only need whether all are unique
        unique_vals, _ = torch.unique(token_indices, return_inverse=True)
        unique_count = unique_vals.numel()

        grid = (M,)  # one program per row i in [0, M)

        if unique_count == M:
            # No duplicates: non-atomic copy is safe and faster
            copy_rows_kernel[grid](
                out, expert_outputs, token_indices,
                M, H,
                num_warps=4,
            )
        else:
            # Duplicates present: use atomic add to accumulate
            scatter_add_atomic_rows_kernel[grid](
                out, expert_outputs, token_indices,
                N, M, H,
                num_warps=4,
            )

        return out


def run(*args):
    return ModelNew()(*args)
