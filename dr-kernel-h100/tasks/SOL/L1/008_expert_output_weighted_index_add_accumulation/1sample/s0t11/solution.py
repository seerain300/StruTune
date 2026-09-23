import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_kernel(
    out_ptr,                       # *fp16
    exp_ptr,                       # *fp16
    token_indices_ptr,             # *int32
    batch_seq_len: tl.constexpr,   # int
    hidden_size: tl.constexpr,     # int
    n_tokens: tl.constexpr,        # int
    stride_row: tl.constexpr,      # int (out.stride(0))
    stride_col: tl.constexpr,      # int (out.stride(1))
    exp_stride_row: tl.constexpr,  # int (exp.stride(0))
    exp_stride_col: tl.constexpr,  # int (exp.stride(1))
    BLOCK_SIZE: tl.constexpr       # int, e.g., 256
):
    pid = tl.program_id(0)  # token id
    if pid >= n_tokens:
        return

    # Load token index (int32)
    idx = tl.load(token_indices_ptr + pid)
    if idx < 0 or idx >= batch_seq_len:
        return

    # Iterate over hidden columns in chunks
    col = 0
    while col < hidden_size:
        cols = col + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size

        # Load expert slice for this token
        src = tl.load(
            exp_ptr + pid * exp_stride_row + cols * exp_stride_col,
            mask=mask,
            other=0.0
        )

        # Destination pointers
        dest_ptrs = out_ptr + idx * stride_row + cols * stride_col

        # Atomic add
        tl.atomic_add(dest_ptrs, src, mask=mask)

        col += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states, expert_outputs, token_indices):
        # Shapes
        batch_size = axes_and_scalars["batch_size"]
        seq_len = axes_and_scalars["seq_len"]
        hidden_size = axes_and_scalars["hidden_size"]
        batch_seq_len = batch_size * seq_len
        n_tokens = expert_outputs.shape[0]

        # Output buffer: zero-initialized for correct accumulation
        # Note: device comes from final_hidden_states
        out = torch.zeros((batch_seq_len, hidden_size), dtype=torch.bfloat16, device=final_hidden_states.device)

        # Ensure inputs are contiguous
        expert_outputs = expert_outputs.contiguous()
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        # Strides (in elements)
        stride_row = out.stride(0)
        stride_col = out.stride(1)
        exp_stride_row = expert_outputs.stride(0)
        exp_stride_col = expert_outputs.stride(1)

        # Launch Triton kernel: one program per token
        grid = (n_tokens,)
        scatter_add_kernel[grid](
            out, expert_outputs, token_indices_i32,
            batch_seq_len=batch_seq_len,
            hidden_size=hidden_size,
            n_tokens=n_tokens,
            stride_row=stride_row,
            stride_col=stride_col,
            exp_stride_row=exp_stride_row,
            exp_stride_col=exp_stride_col,
            BLOCK_SIZE=256,  # tuned for hidden_size=256; generalizes via while loop
            num_warps=4
        )

        return out


def run(*args):
    return ModelNew()(*args)
