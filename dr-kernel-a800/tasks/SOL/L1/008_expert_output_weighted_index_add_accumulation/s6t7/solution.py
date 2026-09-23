import torch
import triton
import triton.language as tl


@triton.jit
def _copy_rows_kernel(out_ptr, src_ptr, rows, H: tl.constexpr, BLOCK_H: tl.constexpr):
    # 2D grid: pid0 over rows, pid1 over tiles of the hidden dimension
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    offs = col_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < H
    # Compute pointers for this row and column offsets
    out_row_ptr = out_ptr + row * H + offs
    src_row_ptr = src_ptr + row * H + offs
    # Load and store with mask to handle tail
    val = tl.load(src_row_ptr, mask=mask, other=0.0)
    tl.store(out_row_ptr, val, mask=mask)


def triton_copy_rows(out: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """
    Copy src into out using a Triton kernel. Assumes out and src have same shape [M, H]
    and dtype. We launch a 2D grid over rows and hidden dimension tiles.
    """
    assert out.shape == src.shape
    assert out.device.type == 'cuda'
    M, H = out.shape
    # Choose a tile size; 128 works well for typical hidden sizes and bfloat16
    BLOCK_H = 128
    grid = (M, triton.cdiv(H, BLOCK_H))
    _copy_rows_kernel[grid](out, src, M, H, BLOCK_H, num_warps=4, num_stages=2)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        # 1) Use Triton to copy final_hidden_states into output (clone semantics)
        output = torch.empty_like(final_hidden_states)
        triton_copy_rows(output, final_hidden_states)

        # 2) Perform scatter-add using PyTorch to exactly match index_add behavior
        # index_add along dim=0: add expert_outputs[i] to output[token_indices[i]]
        output.index_add_(0, token_indices, expert_outputs)

        return output


def run(*args):
    return ModelNew()(*args)
