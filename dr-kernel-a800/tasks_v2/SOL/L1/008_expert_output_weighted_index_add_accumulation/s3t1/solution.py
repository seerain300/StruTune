import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,       # *fp32, shape [batch_seq_len, H]
    A_ptr,         # *fp32, shape [N, H]
    idx_ptr,       # *int32, shape [N]
    N,             # int32, number of updates
    H,             # int32, hidden size
    BLOCK_SIZE: tl.constexpr,  # compile-time constant block size (== H)
):
    pid = tl.program_id(axis=0)  # one program per update
    # Load the destination row index
    idx = tl.load(idx_ptr + pid)  # int32
    # Column offsets for this row
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H  # if BLOCK_SIZE == H, mask is all true

    # Load the expert output row (fp32)
    a_row_ptr = A_ptr + pid * H
    v = tl.load(a_row_ptr + offs, mask=mask, other=0.0)  # fp32 vector

    # Load current output row (fp32)
    out_row_ptr = out_ptr + idx * H
    out_vals = tl.load(out_row_ptr + offs, mask=mask, other=0.0)  # fp32 vector

    # Atomic add into output (equivalent to +=)
    out_vals += v
    tl.store(out_row_ptr + offs, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We perform accumulation in fp32 using atomic adds, then cast back to bfloat16.
        """
        # Ensure tensors are on CUDA and have expected shapes
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."
        assert final_hidden_states.dim() == 2 and expert_outputs.dim() == 2, "final_hidden_states must be [batch_seq_len, H], expert_outputs must be [N, H]."
        batch_seq_len, H = final_hidden_states.shape
        N = expert_outputs.shape[0]
        assert token_indices.numel() == N, "token_indices length must equal number of expert outputs."

        # Make inputs contiguous
        final_hidden_states = final_hidden_states.contiguous()
        token_indices = token_indices.contiguous()

        # Allocate fp32 output buffer for atomic accumulation
        out_fp32 = torch.empty((batch_seq_len, H), dtype=torch.float32, device=final_hidden_states.device)

        # Prepare inputs for Triton: convert expert_outputs to fp32 and token_indices to int32
        A_fp32 = expert_outputs.to(torch.float32).contiguous()
        idx_i32 = token_indices.to(torch.int32).contiguous()

        # Choose BLOCK_SIZE: set to H (compile-time constant for the kernel)
        BLOCK_SIZE = H

        # Launch one program per update
        grid = (N,)
        scatter_add_rows_kernel[grid](
            out_fp32, A_fp32, idx_i32,
            N, H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

        # Cast back to bfloat16 to match the original output dtype
        output = out_fp32.to(torch.bfloat16)
        return output


def run(*args):
    return ModelNew()(*args)
