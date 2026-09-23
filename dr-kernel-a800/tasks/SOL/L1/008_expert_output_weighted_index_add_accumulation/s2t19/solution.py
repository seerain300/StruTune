import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_dim0_chunked_kernel(
    output_ptr,       # *bf16, shape [M, H], contiguous
    indices_ptr,      # *int32, shape [N]
    inputs_ptr,       # *bf16, shape [N, H], contiguous
    N,                # int32
    M,                # int32 (batch_size * seq_len)
    H,                # int32 (hidden_size)
    stride_out_row,   # int64
    stride_out_col,   # int64
    stride_in_row,    # int64
    stride_in_col,    # int64
    BLOCK: tl.constexpr,  # block size along hidden dimension
):
    # One program per row i
    i = tl.program_id(axis=0)
    if i >= N:
        return

    # Load destination row index
    idx = tl.load(indices_ptr + i)  # int32

    # Compute base pointers for this input row i
    in_base = inputs_ptr + i * stride_in_row
    # Load the entire expert vector in chunks of BLOCK
    for h_start in range(0, H, BLOCK):
        col = h_start + tl.arange(0, BLOCK)  # vector of column offsets
        mask = col < H
        vals = tl.load(in_base + col * stride_in_col, mask=mask, other=0.0)  # shape (BLOCK,), bfloat16

        # Compute output pointers for row idx and add vals
        out_base = output_ptr + idx * stride_out_row
        out_ptrs = out_base + col * stride_out_col
        # Perform vectorized atomic add across columns
        tl.atomic_add(out_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Preserve original semantics: clone the accumulator (random init)
        output = final_hidden_states.clone()

        # If not on CUDA, fallback to PyTorch to ensure correctness
        if (not output.is_cuda) or (not expert_outputs.is_cuda) or (not token_indices.is_cuda):
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return output

        # Ensure inputs are contiguous
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.to(torch.int32).contiguous()

        # Shapes
        M, H = output.shape  # batch_seq_len, hidden_size
        N = expert_outputs.shape[0]
        assert expert_outputs.shape[1] == H, "expert_outputs second dimension must match hidden_size"
        assert token_indices.shape[0] == N, "token_indices length must match expert_outputs rows"

        # Strides in elements
        stride_out_row = output.stride(0)
        stride_out_col = output.stride(1)
        stride_in_row = expert_outputs.stride(0)
        stride_in_col = expert_outputs.stride(1)

        # Launch Triton: one program per row
        BLOCK = 128  # tuneable: 128 or 256 are good choices; 128 reduces risk for large H
        grid = (N,)

        scatter_add_dim0_chunked_kernel[grid](
            output, token_indices, expert_outputs,
            N, M, H,
            stride_out_row, stride_out_col,
            stride_in_row, stride_in_col,
            BLOCK=BLOCK,
            num_warps=4,   # tuneable
            num_stages=2   # tuneable
        )

        return output


def run(*args):
    return ModelNew()(*args)
