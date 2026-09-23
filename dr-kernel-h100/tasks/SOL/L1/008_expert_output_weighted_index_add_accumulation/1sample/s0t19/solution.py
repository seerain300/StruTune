import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_atomic_kernel(
    out_ptr,            # *bf16, shape [batch_seq_len, hidden_size]
    stride_row,         # int, stride for rows in out
    stride_col,         # int, stride for cols in out
    expert_ptr,         # *bf16, shape [n_tokens, hidden_size]
    ep_row_stride,      # int, stride for rows in expert_outputs
    ep_col_stride,      # int, stride for cols in expert_outputs
    token_idx_ptr,      # *int32, shape [n_tokens]
    batch_seq_len: tl.constexpr,  # not used directly but can be passed
    hidden_size,        # int
    n_tokens,           # int
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per token
    # If grid > n_tokens, guard with mask
    # We set grid=(n_tokens,) so pid < n_tokens is always true; but keep mask for safety.
    mask_pid = pid < n_tokens

    # Load token index for this program
    tok = tl.load(token_idx_ptr + pid, mask=mask_pid, other=0)
    # Triton pointers assume linear addressing; we index with tok * stride_row

    # Iterate over hidden_size in chunks
    # Note: Triton allows Python-like loops inside @triton.jit, here we use a for-range.
    for col in range(0, hidden_size, BLOCK_SIZE):
        offs = tl.arange(0, BLOCK_SIZE)
        cols = col + offs
        mask = mask_pid & (cols < hidden_size)

        # Load source vector slice for this token
        # expert_outputs is 2D, we need row=pid and columns=cols
        src_ptr = expert_ptr + pid * ep_row_stride + cols * ep_col_stride
        src = tl.load(src_ptr, mask=mask, other=0.0)

        # Compute destination pointer: out[tok, cols]
        dst_ptr = out_ptr + tok * stride_row + cols * stride_col
        # Atomic add into destination
        tl.atomic_add(dst_ptr, src, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Output shape must match final_hidden_states: (batch_seq_len, hidden_size)
        batch_size, seq_len, hidden_size = final_hidden_states.shape
        batch_seq_len = batch_size * seq_len

        # Create output and zero-initialize
        out = torch.empty((batch_seq_len, hidden_size), dtype=torch.bfloat16, device=expert_outputs.device)
        out.zero_()  # reliable zero-init

        # Triton expects int32 for indexing
        token_indices_i32 = token_indices.to(torch.int32).contiguous()
        expert_outputs = expert_outputs.contiguous()

        n_tokens = expert_outputs.shape[0]

        # Launch Triton kernel: one program per token
        BLOCK_SIZE = 128  # tune: 128 or 256
        grid = (n_tokens,)
        scatter_add_atomic_kernel[grid](
            out, out.stride(0), out.stride(1),
            expert_outputs, expert_outputs.stride(0), expert_outputs.stride(1),
            token_indices_i32,
            batch_seq_len, hidden_size, n_tokens,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,  # tune for your GPU
        )

        return out


def run(*args):
    return ModelNew()(*args)
