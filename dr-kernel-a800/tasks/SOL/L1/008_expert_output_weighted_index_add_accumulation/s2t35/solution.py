import torch
import triton
import triton.language as tl


@triton.jit
def _index_add_rows_kernel(
    out_ptr,        # *half (final_hidden_states clone, updated via atomic adds)
    expert_ptr,     # *half (expert_outputs)
    indices_ptr,    # *int32 (token_indices)
    N,              # int32: number of selected tokens
    H,              # int32: hidden size
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Destination row index for this selected token
    dest_idx = tl.load(indices_ptr + pid)
    # Base offset for contiguous row-major layout
    out_base = dest_idx * H

    # Process hidden dimension in BLOCK-sized chunks with masks for tail
    for col_start in range(0, H, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < H

        # Load source vector chunk (row pid)
        src_ptr_row = expert_ptr + pid * H + col_start
        vals = tl.load(src_ptr_row + tl.arange(0, BLOCK), mask=mask, other=0.0)

        # Atomically add into output row
        dst_ptr_row = out_ptr + out_base + col_start + tl.arange(0, BLOCK)
        tl.atomic_add(dst_ptr_row, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure CUDA tensors and expected dtypes
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Triton kernel requires CUDA tensors"
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Expected bfloat16 tensors"
        assert token_indices.dtype == torch.int32, "token_indices must be int32"

        # Clone accumulation buffer
        out = final_hidden_states.clone()

        # Shapes
        N = expert_outputs.shape[0]  # number of selected tokens
        H = expert_outputs.shape[1]  # hidden size

        # Launch configuration: choose BLOCK based on H
        if H <= 64:
            BLOCK = 64
            num_warps = 4
        elif H <= 128:
            BLOCK = 128
            num_warps = 4
        else:
            BLOCK = 256
            num_warps = 8
        num_stages = 2

        # Grid: one program per row
        grid = (N,)

        _index_add_rows_kernel[grid](
            out,
            expert_outputs,
            token_indices,
            N,
            H,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return out


def run(*args):
    return ModelNew()(*args)
