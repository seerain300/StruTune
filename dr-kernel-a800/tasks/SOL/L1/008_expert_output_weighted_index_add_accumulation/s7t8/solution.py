import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_atomic_rowwise(out_ptr,  # *out[b, h] (bfloat16)
                                indices_ptr,  # *indices[N] (int64, but used as int32)
                                src_ptr,      # *src[K, h] (bfloat16)
                                N,           # number of rows in out to write (== sum of contributions)
                                H,           # hidden_size (columns)
                                BLOCK_H: tl.constexpr):
    """
    For each i in [0, N):
        row_idx = indices[i] (int32)
        for j in [0, H):
            out[row_idx, j] += src[i, j]
    Uses atomic_add to handle duplicates in indices.
    """
    pid = tl.program_id(axis=0)
    # Each program handles one row i = pid
    # Guard against out-of-bounds pid (in case grid > N)
    if pid >= N:
        return

    # Load destination row index
    row_idx = tl.load(indices_ptr + pid).to(tl.int32)

    # Iterate over hidden dimension in chunks of BLOCK_H (compile-time constant).
    # We keep it simple and loop j from 0 to H-1 for correctness.
    # Note: Triton supports such scalar loops. For robustness, we avoid vectorized pointer arithmetic.
    for j in range(0, H):
        src_offset = pid * H + j
        dst_offset = row_idx * H + j

        # Load source element and atomic add to destination
        val = tl.load(src_ptr + src_offset)
        # out_ptr and src_ptr are bfloat16; atomic_add supports bf16
        tl.atomic_add(out_ptr + dst_offset, val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        """
        Triton implementation of:
            out = final_hidden_states.clone()
            out.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        # Ensure tensors are on CUDA
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA device for Triton kernel."

        # Clone to match reference behavior exactly
        out = final_hidden_states.clone()

        # Ensure contiguous
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        N = out.shape[0]  # number of rows we may write to (this equals batch_seq_len)
        K = expert_outputs.shape[0]  # number of contributions
        H = out.shape[1]

        # Triton grid: one program per row
        grid = (triton.cdiv(N, 1),)  # grid size equals number of rows; we guard pid >= N in kernel

        # Launch kernel
        scatter_add_atomic_rowwise[grid](
            out,                    # out_ptr
            token_indices,          # indices_ptr
            expert_outputs,         # src_ptr
            N,                      # number of rows in out
            H,                      # hidden size
            BLOCK_H=1              # loop over hidden dimension one by one for robustness
        )

        return out


def run(*args):
    return ModelNew()(*args)
