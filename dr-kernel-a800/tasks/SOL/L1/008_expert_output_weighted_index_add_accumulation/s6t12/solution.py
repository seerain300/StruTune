import torch
import triton
import triton.language as tl


@triton.jit
def copy_rows_kernel(out_ptr, in_ptr, M, N, BLOCK_N: tl.constexpr):
    """
    Row-wise copy: out[i, :] = in[i, :] for i in [0, M).
    Columns processed in chunks of BLOCK_N for any N.
    """
    row = tl.program_id(0)
    # guard: only launch up to M rows
    if row >= M:
        return
    # loop over columns in chunks
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < N
        in_off = row * N + cols
        out_off = row * N + cols
        # Assume inputs/outputs are bf16; Triton load/store will use pointer dtype.
        tl.store(out_ptr + out_off, tl.load(in_ptr + in_off, mask=mask, other=0.0))


@triton.jit
def scatter_add_rows_kernel(out_ptr, src_ptr, idx_ptr, K, N, BLOCK_N: tl.constexpr):
    """
    Per-token scatter-add: for each i in [0, K),
    out[token_indices[i], :] += src[i, :].
    Columns processed in chunks of BLOCK_N for any N.
    """
    pid = tl.program_id(0)  # one program per token
    # Load destination row index
    idx = tl.load(idx_ptr + pid)  # int32
    # Compute row base pointers
    src_row_ptr = src_ptr + pid * N
    dst_row_ptr = out_ptr + idx * N
    # Iterate over N in chunks
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < N
        src_vals = tl.load(src_row_ptr + cols, mask=mask, other=0.0)
        dst_vals = tl.load(dst_row_ptr + cols, mask=mask, other=0.0)
        dst_vals += src_vals
        tl.store(dst_row_ptr + cols, dst_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, token_indices, expert_outputs)
        """
        # Ensure inputs are on CUDA and bf16. The original setup returns bf16 tensors.
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA"
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Dtype must be bfloat16"

        M, N = final_hidden_states.shape  # [batch_seq_len, hidden_size]
        K = expert_outputs.shape[0]

        # Step 1: Row-wise copy from final_hidden_states to output (clone semantics).
        output = torch.empty((M, N), dtype=final_hidden_states.dtype, device=final_hidden_states.device)
        grid_copy = (M,)
        # Choose BLOCK_N based on N; keep modest to balance occupancy and bandwidth.
        BLOCK_N = 128 if N >= 128 else 64
        copy_rows_kernel[grid_copy](output, final_hidden_states, M, N, BLOCK_N=BLOCK_N)

        # Step 2: Per-token scatter-add into output
        # Ensure token_indices are int32 for Triton
        idx32 = token_indices.to(torch.int32)

        grid_add = (K,)
        scatter_add_rows_kernel[grid_add](output, expert_outputs, idx32, K, N, BLOCK_N=BLOCK_N)

        return output


def run(*args):
    return ModelNew()(*args)
