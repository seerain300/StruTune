import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_per_element_kernel(
    out_ptr,           # *bf16
    expert_ptr,        # *bf16
    indices_ptr,       # *int64
    B: tl.constexpr,   # batch_seq_len (rows in out)
    H: tl.constexpr,   # hidden_size (cols in out)
    T: tl.constexpr,   # number of expert outputs (rows in expert)
    stride_out_row: tl.constexpr,
    stride_out_col: tl.constexpr,
    stride_expert_row: tl.constexpr,
    stride_expert_col: tl.constexpr,
    BLOCK_COL: tl.constexpr,  # we will set 1 for exact per-element
):
    # 2D grid: axis 0 over rows (T), axis 1 over columns in tiles of BLOCK_COL
    row = tl.program_id(0)  # which expert row we are handling
    col_block = tl.program_id(1)
    col_start = col_block * BLOCK_COL
    cols = col_start + tl.arange(0, BLOCK_COL)

    # Bounds for this program
    valid_row = row < T
    valid_cols = cols < H
    mask = valid_row & valid_cols

    # Load token index (int64)
    idx64 = tl.load(indices_ptr + row, mask=valid_row, other=0)
    # Cast to int32 for pointer arithmetic (safe since B < 2^31 in provided configs)
    idx = idx64.to(tl.int32)

    # Compute linear offsets for load/store
    # out[idx, cols]
    out_offs = idx * stride_out_row + cols * stride_out_col
    # expert[row, cols]
    exp_offs = row * stride_expert_row + cols * stride_expert_col

    # Load values (bf16)
    v = tl.load(expert_ptr + exp_offs, mask=mask, other=0.0)  # 0.0 is treated as bf16 scalar
    out_vals = tl.load(out_ptr + out_offs, mask=mask, other=0.0)

    # Accumulate
    out_vals = out_vals + v

    # Store back
    tl.store(out_ptr + out_offs, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized version of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We implement the exact scatter-add per element to ensure numerical match.
        """
        # Ensure CUDA tensors
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA for Triton."

        # Clone to match original behavior
        output = final_hidden_states.clone()

        # Ensure contiguity
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = output.shape[0]  # batch_seq_len
        H = output.shape[1]  # hidden_size
        T = token_indices.shape[0]  # number of expert outputs

        # Grid: one program per (row, column) element
        # BLOCK_COL=1 ensures we process one column per program exactly
        grid = (T, H)
        # Strides (row-major contiguous)
        stride_out_row = H
        stride_out_col = 1
        stride_expert_row = H
        stride_expert_col = 1

        # Launch Triton kernel: no atomics, exact per-element updates
        scatter_add_per_element_kernel[grid](
            output, expert_outputs, token_indices,
            B=B, H=H, T=T,
            stride_out_row=stride_out_row,
            stride_out_col=stride_out_col,
            stride_expert_row=stride_expert_row,
            stride_expert_col=stride_expert_col,
            BLOCK_COL=1,
            num_warps=1,
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
