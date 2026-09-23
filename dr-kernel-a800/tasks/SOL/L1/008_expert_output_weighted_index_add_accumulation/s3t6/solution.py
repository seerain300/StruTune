import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_fp32_ptr,  # *fp32, shape [batch_seq_len, H]
    A_fp32_ptr,    # *fp32, shape [N, H]
    idx_ptr,       # *int32, shape [N]
    N,             # number of updates (num_selected_tokens)
    H: tl.constexpr,  # hidden size (compile-time constant for tiling)
    BLOCK_SIZE: tl.constexpr,  # tile size along hidden dimension
):
    # One program per update
    i = tl.program_id(0)
    if i >= N:
        return

    # Load destination row index
    idx = tl.load(idx_ptr + i)  # int32
    # Compute base offsets for the row in out and for the vector in A
    out_row_base = idx * H
    A_row_base = i * H

    # Process the hidden vector in tiles of size BLOCK_SIZE
    for offset in range(0, H, BLOCK_SIZE):
        offs = tl.arange(0, BLOCK_SIZE)
        col = offset + offs  # vector of column indices
        mask = col < H

        # Load destination row slice and source slice
        out_vals = tl.load(out_fp32_ptr + out_row_base + col, mask=mask, other=0.0)
        v = tl.load(A_fp32_ptr + A_row_base + col, mask=mask, other=0.0)

        # Add and store back
        out_vals += v
        tl.store(out_fp32_ptr + out_row_base + col, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We compute this using a Triton kernel that performs load+add+store per update in fp32, then cast to bfloat16.
        """
        # Ensure on CUDA and contiguous
        assert final_hidden_states.is_cuda, "Inputs must be on CUDA device for Triton kernel."
        assert expert_outputs.is_cuda, "Inputs must be on CUDA device for Triton kernel."
        assert token_indices.is_cuda, "Inputs must be on CUDA device for Triton kernel."

        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        batch_seq_len, H = final_hidden_states.shape
        N = expert_outputs.shape[0]
        assert token_indices.numel() == N, "token_indices length must equal number of expert outputs."
        assert token_indices.dtype in (torch.int64, torch.int32), "token_indices must be long or int32."
        assert final_hidden_states.dtype == torch.bfloat16, "final_hidden_states must be bfloat16 as per get_inputs."
        assert expert_outputs.dtype == torch.bfloat16, "expert_outputs must be bfloat16 as per get_inputs."

        # Allocate fp32 output buffer initialized to match final_hidden_states (clone semantics)
        out_fp32 = final_hidden_states.to(torch.float32).clone()

        # Prepare inputs for Triton: convert to fp32 and int32
        A_fp32 = expert_outputs.to(torch.float32)
        idx_i32 = token_indices.to(torch.int32)

        # Grid: one program per update
        grid = (N,)

        # Choose a BLOCK_SIZE. Using H is fine since H is constexpr in kernel; masking handles cases where H is not multiple of BLOCK_SIZE.
        # Keep BLOCK_SIZE modest (e.g., 128 or 256) to balance occupancy and simplicity. Given typical hidden sizes in these workloads, 128 is fine.
        BLOCK_SIZE = 128

        scatter_add_rows_kernel[grid](
            out_fp32, A_fp32, idx_i32,
            N, H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

        # Cast back to bfloat16 to match original output dtype
        output = out_fp32.to(torch.bfloat16)
        return output


def run(*args):
    return ModelNew()(*args)
