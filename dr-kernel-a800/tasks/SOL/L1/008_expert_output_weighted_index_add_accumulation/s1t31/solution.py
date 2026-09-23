import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,    shape [N]
    M: tl.constexpr,  # total rows in out_ptr/src_ptr (batch_seq_len)
    N: tl.constexpr,  # total source rows (batch_seq_len * num_experts_per_tok)
    H: tl.constexpr,  # hidden size
    BLOCK_H: tl.constexpr,  # tile size for H (e.g., 256)
):
    # Each program handles one source row 'pid' in [0, N)
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Load token index for this source row (int32)
    index = tl.load(indices_ptr + pid)

    # Iterate over H in chunks of BLOCK_H; typical H <= 1024 so this runs once
    offs = tl.arange(0, BLOCK_H)
    for start in range(0, H, BLOCK_H):
        h_offsets = start + offs
        mask = h_offsets < H

        # Load source row segment (bfloat16)
        src_row_ptr = src_ptr + pid * H + h_offsets
        src_vals = tl.load(src_row_ptr, mask=mask, other=0.0)

        # Compute destination row pointer
        out_row_ptr = out_ptr + index * H + h_offsets

        # Atomic add into output
        tl.atomic_add(out_row_ptr, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add along dim=0:
          output[i] += expert_outputs[j] for each j where token_indices[j] == i
        Requires CUDA tensors. The output tensor is modified in-place and returned.
        """
        # Ensure CUDA and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Triton requires CUDA tensors"
        out = final_hidden_states.clone()

        # Shapes
        M = out.shape[0]  # batch_seq_len
        N = expert_outputs.shape[0]
        H = out.shape[1]

        # Ensure contiguity and dtypes
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous().to(torch.int32)

        # Launch Triton kernel: one program per source row
        grid = (N,)
        BLOCK_H = 256
        scatter_add_rows_kernel[grid](
            out, expert_outputs, token_indices,
            M, N, H,
            BLOCK_H,
            num_warps=4,
            num_stages=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
