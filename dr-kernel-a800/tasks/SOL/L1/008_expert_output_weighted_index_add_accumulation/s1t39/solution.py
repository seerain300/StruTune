import torch
import triton
import triton.language as tl


@triton.jit
def direct_scatter_rows_kernel(
    out_ptr,          # *bfloat16, shape [M, H]
    src_ptr,          # *bfloat16, shape [N, H]
    indices_ptr,      # *int32,    shape [N]
    M: tl.constexpr,  # total rows in output
    H: tl.constexpr,  # hidden size
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # Each program handles one source row i
    i = tl.program_id(0)
    if i >= N:
        return

    # Load destination row index for this source row
    row_idx = tl.load(indices_ptr + i)  # int32
    if row_idx < 0 or row_idx >= M:
        return  # defensive, though indices should be valid

    # Iterate across hidden dimension in tiles
    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask = h_offsets < H

        # Load source row segment
        src_row_ptrs = src_ptr + i * H + h_offsets
        src_vals = tl.load(src_row_ptrs, mask=mask, other=0.0)

        # Compute output row segment pointers and store
        out_row_ptrs = out_ptr + row_idx * H + h_offsets
        tl.store(out_row_ptrs, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        """
        Triton-only implementation of:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        # Ensure device consistency and contiguity
        assert final_hidden_states.is_cuda, "final_hidden_states must be on CUDA"
        assert expert_outputs.is_cuda, "expert_outputs must be on CUDA"
        assert token_indices.is_cuda, "token_indices must be on CUDA"

        M = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]

        # Clone to match original behavior exactly (before scatter)
        output = final_hidden_states.clone()

        # Ensure dtypes and contiguity
        # expert_outputs and output should be bfloat16 as per get_inputs
        # token_indices should be int32
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)
        if not output.is_contiguous():
            output = output.contiguous()
        if not expert_outputs.is_contiguous():
            expert_outputs = expert_outputs.contiguous()

        # Launch Triton kernel: one program per source row
        BLOCK_H = 256  # tile across H, good balance for H up to 1024
        grid = (N,)

        direct_scatter_rows_kernel[grid](
            output, expert_outputs, token_indices,
            M, H,
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
