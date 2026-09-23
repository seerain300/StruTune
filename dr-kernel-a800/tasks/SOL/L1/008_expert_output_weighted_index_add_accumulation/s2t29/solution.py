import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_add_rows_kernel(
    out_ptr,       # *bf16, pointer to (M, H) output
    in_ptr,        # *bf16, pointer to (N, H) expert_outputs
    idx_ptr,       # *int32, pointer to token_indices (int32)
    M,             # int32: number of rows in output (batch_size * seq_len)
    H,             # int32: hidden size (number of columns)
    BLOCK: tl.constexpr,  # chunk size along hidden dimension
):
    pid = tl.program_id(axis=0)  # one program per row i in [0, N)
    # Optional safety: if pid >= N, early return. In our grid setup, pid < N always.
    # Compute the destination row index for this program
    dest_row = tl.load(idx_ptr + pid)  # int32

    # Loop over hidden dimension in chunks of BLOCK
    # We vectorize across columns and use masked atomic_add for the tail.
    for col_start in range(0, H, BLOCK):
        cols = col_start + tl.arange(0, BLOCK)  # [BLOCK] vector of column indices
        mask = cols < H

        # Load the expert vector slice for this row i
        in_off = pid * H + cols
        vals = tl.load(in_ptr + in_off, mask=mask, other=0.0)

        # Compute output offsets for the destination row and perform atomic add
        out_off = dest_row * H + cols
        tl.atomic_add(out_ptr + out_off, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure all tensors are on the same device and dtype
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA"
        # Ensure dtypes: original code uses bfloat16; keep consistent
        if final_hidden_states.dtype != expert_outputs.dtype:
            expert_outputs = expert_outputs.to(final_hidden_states.dtype)
        # Make tensors contiguous to avoid stride-related issues
        out = final_hidden_states.clone().contiguous()  # we will modify this in-place via index_add semantics
        # Note: We will instead run the Triton kernel directly on out
        in_t = expert_outputs.contiguous()
        idx_t = token_indices.to(torch.int32).contiguous()

        # Shapes
        M = out.shape[0]                  # batch_size * seq_len
        H = out.shape[1]                  # hidden_size

        # Launch one program per selected token (row in expert_outputs)
        N = expert_outputs.shape[0]
        grid = (N,)

        # Use fixed, robust launch params that previously achieved best speed
        _scatter_add_rows_kernel[grid](
            out, in_t, idx_t,
            M, H,
            BLOCK=128,
            num_warps=4,
            num_stages=2,
        )

        # 'out' now contains the result of index_add along dim=0
        return out


def run(*args):
    return ModelNew()(*args)
