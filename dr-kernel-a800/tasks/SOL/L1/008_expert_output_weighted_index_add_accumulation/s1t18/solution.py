import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,    shape [N]
    M,                # int32, total rows in out_ptr/src_ptr
    H,                # int32, hidden size
    BLOCK_H: tl.constexpr,  # tile size along H (e.g., 256)
):
    pid = tl.program_id(axis=0)  # program id: row index in [0, N)
    # Compute pointers for this source row
    src_row_ptr = src_ptr + pid * H
    # Load token index for this row
    out_row_index = tl.load(indices_ptr + pid)  # token position in [0, M)
    out_row_ptr = out_ptr + out_row_index * H

    # Iterate over H in tiles of BLOCK_H
    for j in range(0, H, BLOCK_H):
        offsets = j + tl.arange(0, BLOCK_H)
        mask = offsets < H
        # Load source values (masked for tail)
        vals = tl.load(src_row_ptr + offsets, mask=mask, other=0.0)
        # Atomic add into output at token_indices[pid]
        tl.atomic_add(out_row_ptr + offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure inputs are CUDA tensors and have expected dtypes
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA"
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Use bfloat16 tensors"
        assert token_indices.dtype == torch.int64, "token_indices must be torch.long (int64)"
        M = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]
        # Output buffer (clone of final_hidden_states)
        output = final_hidden_states.clone()

        # Ensure tensors are contiguous
        expert_outputs = expert_outputs.contiguous()
        token_indices_i32 = token_indices.to(torch.int32)

        # Fixed tile size for balanced performance across common H
        BLOCK_H = 256
        num_warps = 4  # good balance for 256-wide vectors
        num_stages = 2

        # Launch one program per source row
        grid = (N,)
        scatter_add_rows_kernel[grid](
            output, expert_outputs, token_indices_i32,
            M, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return output


def run(*args):
    return ModelNew()(*args)
