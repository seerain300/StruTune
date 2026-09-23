import torch

# Triton is required for the custom kernel
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def scatter_add_rows_kernel(
    out_ptr, A_ptr, idx_ptr,
    N, H
):
    # Single program processes all tokens and hidden elements in nested loops
    # out_ptr: [B, H] float32
    # A_ptr:   [N, H] float32
    # idx_ptr: [N] int32
    for i in range(0, N):
        dest_row = tl.load(idx_ptr + i)  # int32
        # For each hidden dimension j, add A[i, j] to out[dest_row, j]
        for j in range(0, H):
            # out pointer offset for the element (dest_row, j)
            out_offset = dest_row * H + j
            # A pointer offset for the element (i, j)
            A_offset = i * H + j
            # Load fp32 values and add
            val = tl.load(out_ptr + out_offset)  # fp32
            add_val = tl.load(A_ptr + A_offset)  # fp32
            val = val + add_val
            tl.store(out_ptr + out_offset, val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        """
        Triton implementation of:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We accumulate in fp32 using a simple nested loop kernel and cast back to bfloat16.
        """
        # Ensure inputs are contiguous
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        batch_seq_len, H = final_hidden_states.shape
        N = expert_outputs.shape[0]
        assert token_indices.numel() == N, "token_indices length must equal number of expert outputs."

        # Allocate fp32 output buffer for accumulation (clone fp32 view for correctness)
        out_fp32 = final_hidden_states.to(torch.float32).contiguous()

        # Prepare inputs: expert_outputs as fp32, token_indices as int32
        A_fp32 = expert_outputs.to(torch.float32)
        idx_i32 = token_indices.to(torch.int32)

        # Launch a single program; nested loops inside the kernel will handle all tokens and hidden elements
        grid = (1,)

        scatter_add_rows_kernel[grid](
            out_fp32, A_fp32, idx_i32,
            N, H,
            num_warps=1,
        )

        # Cast back to bfloat16 to match original output dtype
        output = out_fp32.to(torch.bfloat16)
        return output


def run(*args):
    return ModelNew()(*args)
