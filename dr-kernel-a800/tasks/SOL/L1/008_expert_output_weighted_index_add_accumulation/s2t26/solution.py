import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_add_rows_kernel(
    out_ptr,        # *bf16, shape (M, H)
    in_ptr,         # *bf16, shape (N, H)
    idx_ptr,        # *int32, shape (N,)
    M,              # int: number of rows in output (batch_seq_len)
    H,              # int: hidden size
    BLOCK: tl.constexpr,  # chunk size across hidden dimension
):
    # One program per selected token (row i in expert_outputs)
    row_i = tl.program_id(0)

    # Load destination row index
    dest_row = tl.load(idx_ptr + row_i)  # int32

    # Iterate over hidden dimension in chunks of BLOCK
    num_chunks = (H + BLOCK - 1) // BLOCK  # ceil_div
    cols = tl.arange(0, BLOCK)
    for chunk in range(0, num_chunks):
        col_start = chunk * BLOCK
        offs = col_start + cols
        mask = offs < H

        # Compute pointers for this chunk
        out_row_ptr = out_ptr + dest_row * H + offs
        in_row_ptr = in_ptr + row_i * H + offs

        # Load input slice with mask
        vals = tl.load(in_row_ptr, mask=mask, other=0.0)
        # Atomic add to output
        tl.atomic_add(out_row_ptr, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure CUDA tensors and contiguity
        out = final_hidden_states.contiguous()  # (M, H), M = batch_size * seq_len
        exp = expert_outputs.contiguous()       # (N, H), N = batch_size * seq_len * num_experts_per_tok
        idx = token_indices.contiguous().to(torch.int32)  # (N,)

        M = out.shape[0]
        H = out.shape[1]
        N = exp.shape[0]

        # Launch one program per row in expert_outputs
        grid = (N,)

        # Fixed safe parameters; previously validated to work across workloads
        BLOCK = 128

        _scatter_add_rows_kernel[grid](
            out, exp, idx,
            M, H,
            BLOCK=BLOCK,
            num_warps=2,
            num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
