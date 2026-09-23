import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per token, loop over hidden columns in chunks and atomic_add
@triton.jit
def scatter_add_tokens_kernel(
    out_ptr,              # *bf16
    expert_ptr,           # *bf16
    indices_ptr,          # *int32
    batch_seq_len,        # int
    hidden_size,          # int
    n_tokens,             # int
    out_stride_row,       # int
    out_stride_col,       # int
    exp_stride_row,       # int
    exp_stride_col,       # int
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one token
    pid = tl.program_id(axis=0)
    if pid >= n_tokens:
        return

    # Compute the token index for this program
    # indices_ptr[pid] is int32
    token_idx = tl.load(indices_ptr + pid)

    # Iterate over hidden columns in chunks
    offs = 0
    while offs < hidden_size:
        cols = offs + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size

        # Load the corresponding expert slice
        expert_row_ptr = expert_ptr + pid * exp_stride_row
        expert_vals = tl.load(expert_row_ptr + cols * exp_stride_col, mask=mask, other=0.0)

        # Compute output row pointer
        out_row_ptr = out_ptr + token_idx * out_stride_row

        # Atomic add the slice into the output row
        tl.atomic_add(out_row_ptr + cols * out_stride_col, expert_vals, mask=mask)

        offs += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(
        self,
        final_hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        token_indices: torch.Tensor,
    ):
        # batch_seq_len is inferred from token_indices being within [0, batch_seq_len)
        # but final_hidden_states shape is (batch_seq_len, hidden_size)
        batch_seq_len = final_hidden_states.shape[0]
        hidden_size = final_hidden_states.shape[1]
        n_tokens = expert_outputs.shape[0]

        # Allocate and zero-initialize output
        out = torch.zeros((batch_seq_len, hidden_size), dtype=torch.bfloat16, device=final_hidden_states.device)

        # Ensure inputs are contiguous and dtype/device correct
        expert_outputs = expert_outputs.contiguous()
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        # Strides (in elements)
        out_stride_row, out_stride_col = out.stride(0), out.stride(1)
        exp_stride_row, exp_stride_col = expert_outputs.stride(0), expert_outputs.stride(1)

        # Launch Triton kernel: one program per token
        if TRITON_AVAILABLE:
            grid = (n_tokens,)
            scatter_add_tokens_kernel[grid](
                out,
                expert_outputs,
                token_indices_i32,
                batch_seq_len,
                hidden_size,
                n_tokens,
                out_stride_row,
                out_stride_col,
                exp_stride_row,
                exp_stride_col,
                BLOCK_SIZE=256,   # tuneable: 128 or 256
                num_warps=4,
            )
        else:
            # Fallback: PyTorch implementation if Triton is not available
            out.index_add_(dim=0, index=token_indices, source=expert_outputs)

        return out


def run(*args):
    return ModelNew()(*args)
