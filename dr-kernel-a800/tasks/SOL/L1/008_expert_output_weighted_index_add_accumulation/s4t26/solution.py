import torch
import triton
import triton.language as tl


@triton.jit
def _copy_rows_kernel(output_ptr, src_ptr,
                      B: tl.constexpr, H: tl.constexpr,
                      stride_out_row, stride_out_col,
                      stride_src_row, stride_src_col,
                      BLOCK_H: tl.constexpr):
    # 2D grid: rows x column tiles
    row = tl.program_id(0)
    col_tile = tl.program_id(1)
    offs = col_tile * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < H

    out_ptrs = output_ptr + row * stride_out_row + offs * stride_out_col
    src_ptrs = src_ptr + row * stride_src_row + offs * stride_src_col

    vals = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, vals, mask=mask)


@triton.jit
def _scatter_add_rows_kernel(output_ptr, expert_ptr, indices_ptr,
                             T: tl.constexpr, H: tl.constexpr,
                             stride_out_row, stride_out_col,
                             stride_exp_row, stride_exp_col,
                             BLOCK_H: tl.constexpr):
    # 2D grid: (row in expert_outputs, column tiles)
    row = tl.program_id(0)
    col_tile = tl.program_id(1)
    offs = col_tile * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < H

    # Load token index for this source row
    idx64 = tl.load(indices_ptr + row)
    idx = idx64.to(tl.int32)  # convert to 32-bit for pointer arithmetic

    out_ptrs = output_ptr + idx * stride_out_row + offs * stride_out_col
    exp_ptrs = expert_ptr + row * stride_exp_row + offs * stride_exp_col

    vals = tl.load(exp_ptrs, mask=mask, other=0.0)
    # Elementwise addition (no atomics)
    tl.store(out_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized version of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        All computation is performed via Triton kernels. ModelNew.forward must launch these kernels.
        """
        # Ensure tensors are on CUDA
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."

        # Shapes
        B = final_hidden_states.shape[0]  # number of rows to scatter into
        H = final_hidden_states.shape[1]  # hidden size
        T = token_indices.shape[0]        # number of expert outputs

        # Ensure contiguity for predictable strides
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Allocate output
        output = torch.empty_like(final_hidden_states)

        # 1) Copy final_hidden_states into output using Triton
        BLOCK_H = 128  # tile size along hidden dimension
        grid_copy = (B, triton.cdiv(H, BLOCK_H))
        _copy_rows_kernel[grid_copy](
            output, final_hidden_states,
            B=B, H=H,
            stride_out_row=output.stride(0), stride_out_col=output.stride(1),
            stride_src_row=final_hidden_states.stride(0), stride_src_col=final_hidden_states.stride(1),
            BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        # 2) Scatter-add: output[token_indices[i]] += expert_outputs[i] for each i
        grid_add = (T, triton.cdiv(H, BLOCK_H))
        _scatter_add_rows_kernel[grid_add](
            output, expert_outputs, token_indices,
            T=T, H=H,
            stride_out_row=output.stride(0), stride_out_col=output.stride(1),
            stride_exp_row=expert_outputs.stride(0), stride_exp_col=expert_outputs.stride(1),
            BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
