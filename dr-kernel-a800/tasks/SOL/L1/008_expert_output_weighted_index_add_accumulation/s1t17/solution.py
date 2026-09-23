import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,    shape [N]
    M: tl.constexpr,  # total rows in out_ptr/src_ptr
    H: tl.constexpr,  # hidden size
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # One program per source row
    row = tl.program_id(0)  # 0 <= row < N
    # Load the destination row index (token position)
    dest_idx = tl.load(indices_ptr + row)  # int32

    # Loop over hidden dimension in tiles
    # Triton will unroll this small loop since H is constexpr
    for j in range(0, H, BLOCK_H):
        offsets = j + tl.arange(0, BLOCK_H)  # vector [0..BLOCK_H)
        mask = offsets < H

        # Compute base pointers
        # src_row_ptr: *const bfloat16, length H, starts at src_ptr + row * H
        src_row_ptr = src_ptr + row * H
        out_row_ptr = out_ptr + dest_idx * H

        # Load vector from source row
        vals = tl.load(src_row_ptr + offsets, mask=mask, other=0.0)  # bfloat16

        # Atomic add into output at the selected destination row
        tl.atomic_add(out_row_ptr + offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor
    ) -> torch.Tensor:
        # Clone to initialize output with the same values as final_hidden_states
        output = final_hidden_states.clone()

        # Ensure dtypes/devices/contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA for Triton."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Expected bfloat16 dtype."
        # Make inputs contiguous
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous().to(torch.int32)

        M = output.shape[0]  # batch_seq_len
        H = output.shape[1]
        N = expert_outputs.shape[0]

        # Grid: one program per source row
        grid = (N,)

        # Launch Triton kernel; tune for good performance
        scatter_add_rows_kernel[grid](
            output, expert_outputs, token_indices,
            M, H,
            BLOCK_H=256,
            num_warps=4,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
