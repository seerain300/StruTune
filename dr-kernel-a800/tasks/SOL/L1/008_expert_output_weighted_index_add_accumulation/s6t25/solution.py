import torch
import triton
import triton.language as tl


@triton.jit
def row_copy_kernel(
    src_ptr,  # *const bfloat16
    dst_ptr,  # *bfloat16
    B: tl.constexpr,   # number of rows (batch_seq_len)
    H: tl.constexpr,   # hidden size
    BLOCK_ROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 2D grid over rows and hidden dimension
    pid_rows = tl.program_id(0)
    pid_cols = tl.program_id(1)

    # Compute row and column indices for this program
    row_offsets = pid_rows * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = pid_cols * BLOCK_H + tl.arange(0, BLOCK_H)

    rows = row_offsets[:, None]  # shape (BLOCK_ROWS, 1)
    cols = col_offsets[None, :]  # shape (1, BLOCK_H)

    # Bounds mask for tiles that go beyond B or H
    in_bounds = (rows < B) & (cols < H)  # tl.int1 mask

    # Flat indices for row-major contiguous tensors (stride_row = H, stride_col = 1)
    src_idx = rows * H + cols
    dst_idx = src_idx

    # Load and store with mask; use a bfloat16 zero for masked lanes
    zero = tl.zeros((BLOCK_ROWS, BLOCK_H), dtype=tl.bfloat16)
    src_vals = tl.load(src_ptr + src_idx, mask=in_bounds, other=zero)
    tl.store(dst_ptr + dst_idx, src_vals, mask=in_bounds)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure all inputs are on CUDA and bfloat16
        if not (final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda):
            raise RuntimeError("All inputs must be CUDA tensors for Triton kernels.")
        if final_hidden_states.dtype != torch.bfloat16 or expert_outputs.dtype != torch.bfloat16:
            raise RuntimeError("This implementation expects bfloat16 tensors for final_hidden_states and expert_outputs.")

        B = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]

        # Output starts as a copy of final_hidden_states (clone semantics)
        output = torch.empty_like(final_hidden_states)

        # Choose tile sizes based on hidden size
        if H >= 1024:
            BLOCK_H = 512
        elif H >= 512:
            BLOCK_H = 256
        else:
            BLOCK_H = 128

        BLOCK_ROWS = 128  # process more rows per program to reduce launch overhead

        grid = (triton.cdiv(B, BLOCK_ROWS), triton.cdiv(H, BLOCK_H))

        row_copy_kernel[grid](
            final_hidden_states,
            output,
            B=B,
            H=H,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=2,
        )

        # Accumulate expert_outputs into corresponding token positions (dim=0)
        output.index_add_(0, token_indices, expert_outputs)

        return output


# The evaluator-provided get_inputs (unchanged), used to generate inputs
def get_inputs(axes_and_scalars: dict[str, ...], device: torch.device) -> dict[str, torch.Tensor]:
    """Returns the input arguments for the reference forward pass. Required method."""
    batch_size, seq_len, hidden_size = (
        axes_and_scalars["batch_size"],
        axes_and_scalars["seq_len"],
        axes_and_scalars["hidden_size"],
    )
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]

    batch_seq_len = batch_size * seq_len
    num_selected_tokens = batch_size * seq_len * num_experts_per_tok

    # Initialize accumulation buffer with random values (not zeros)
    final_hidden_states = torch.randn(batch_seq_len, hidden_size, dtype=torch.bfloat16, device=device)

    # Expert outputs
    expert_outputs = torch.randn(num_selected_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    # Token indices
    token_indices = torch.randint(
        0, batch_seq_len, (num_selected_tokens,), dtype=torch.long, device=device
    )

    return {
        "final_hidden_states": final_hidden_states,
        "expert_outputs": expert_outputs,
        "token_indices": token_indices,
    }


def run(*args):
    return ModelNew()(*args)
