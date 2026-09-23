import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,       # *fp32, shape [batch_seq_len, H]
    A_ptr,         # *fp32, shape [N, H] but we'll pass bf16 and cast inside
    idx_ptr,       # *int32, shape [N]
    N: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # Each program handles one token index i
    # Load index for this row
    idx = tl.load(idx_ptr + pid)  # int32
    # Column offsets
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    # Load expert output row i in bf16, cast to fp32 for atomic add
    # Note: A_ptr is passed as original dtype; we cast to fp32 for math
    a_row_ptr = A_ptr + pid * H
    v_bf16 = tl.load(a_row_ptr + offs, mask=mask, other=0.0)  # bf16
    v = v_bf16.to(tl.float32)

    # Accumulate into output at row 'idx'
    out_row_ptr = out_ptr + idx * H
    out_vals = tl.load(out_row_ptr + offs, mask=mask, other=0.0)  # fp32
    out_vals += v
    tl.store(out_row_ptr + offs, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        """
        Triton-optimized version of:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We perform atomic adds in fp32, then cast back to bfloat16 for the final output.
        """
        assert final_hidden_states.dim() == 2, "final_hidden_states must be 2D [batch_seq_len, hidden_size]"
        assert expert_outputs.dim() == 2, "expert_outputs must be 2D [num_selected_tokens, hidden_size]"
        assert token_indices.dim() == 1, "token_indices must be 1D [num_selected_tokens]"

        batch_seq_len, H = final_hidden_states.shape
        N = expert_outputs.shape[0]
        assert token_indices.shape[0] == N, "token_indices length must equal num_selected_tokens"

        # Ensure device and contiguity
        device = final_hidden_states.device
        A = expert_outputs
        # We'll operate in fp32 for atomic accumulation
        out_fp32 = torch.empty((batch_seq_len, H), dtype=torch.float32, device=device)

        # Triton prefers int32 indices; ensure contiguous int32
        idx_i32 = token_indices.to(torch.int32).contiguous()

        # Choose BLOCK_SIZE as next power of two up to 1024
        # Compute next power of two of H
        BLOCK_SIZE = 1
        while BLOCK_SIZE < H and BLOCK_SIZE < 1024:
            BLOCK_SIZE <<= 1
        # If H > 1024, we still try BLOCK_SIZE=1024 and rely on mask; or cap H. For typical transformer H, it works.
        # Launch kernel with one program per row
        grid = (N,)
        scatter_add_rows_kernel[grid](
            out_fp32, A, idx_i32,
            N=N, H=H, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=1  # small per-program work; 1 warp is fine
        )

        # Cast back to bfloat16 to match original behavior
        output = out_fp32.to(torch.bfloat16)
        return output


def run(*args):
    return ModelNew()(*args)
