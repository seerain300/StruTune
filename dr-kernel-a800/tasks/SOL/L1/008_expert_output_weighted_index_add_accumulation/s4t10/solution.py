import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: for each source row i, add expert_outputs[i, h:h+BLOCK_H]
# into output[token_indices[i], h:h+BLOCK_H] for h in [0, H) in BLOCK_H chunks.
@triton.jit
def scatter_add_rows_blocked_kernel(
    output_ptr,         # *bfloat16, shape [B, H]
    expert_ptr,         # *bfloat16, shape [T, H]
    indices_ptr,        # *int64,    shape [T]
    B: tl.int32,        # batch_seq_len (rows in output)
    H: tl.int32,        # hidden_size (cols)
    T: tl.int32,        # num_selected_tokens
    BLOCK_H: tl.constexpr,
):
    # One program per source row
    i = tl.program_id(0)
    if i >= T:
        return

    # Load token index for this source row (int64), then int32 for pointer math
    idx64 = tl.load(indices_ptr + i)
    idx = idx64.to(tl.int32)
    if (idx < 0) or (idx >= B):
        return

    # Iterate over hidden columns in BLOCK_H chunks
    h_start = 0
    while h_start < H:
        offs = tl.arange(0, BLOCK_H)
        h = h_start + offs
        mask = h < H

        # Load source values: expert_outputs[i, h]
        # Row-major: offset = i * H + h
        src_offs = i * H + h
        val = tl.load(expert_ptr + src_offs, mask=mask, other=0.0, dtype=tl.bfloat16)

        # Compute destination offsets: output[idx, h]
        dst_offs = idx * H + h

        # Accumulate: output[idx, h] += val
        out_vals = tl.load(output_ptr + dst_offs, mask=mask, other=0.0, dtype=tl.bfloat16)
        out_vals = out_vals + val
        tl.store(output_ptr + dst_offs, out_vals, mask=mask)

        h_start += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        """
        Triton implementation that performs:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We avoid atomics and process hidden columns in BLOCK_H chunks per program.
        """
        # Clone to match original behavior
        output = final_hidden_states.clone()

        # Ensure contiguity
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = output.shape[0]  # batch_seq_len
        H = output.shape[1]  # hidden_size
        T = token_indices.shape[0]  # number of expert outputs

        # Launch one program per source row; handle H in BLOCK_H chunks inside the kernel
        grid = (T,)
        # Choose a BLOCK_H that works well for typical H (e.g., 128). Use a loop to cover all H.
        # If H is larger, the while loop handles it; for small H, it runs once.
        BLOCK_H = 128

        # Run Triton kernel (if available). Otherwise fallback to PyTorch index_add.
        if TRITON_AVAILABLE and output.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda:
            scatter_add_rows_blocked_kernel[grid](
                output, expert_outputs, token_indices,
                B=B, H=H, T=T,
                BLOCK_H=BLOCK_H,
                num_warps=4,
                num_stages=2,
            )
        else:
            # Fallback: use PyTorch to ensure correctness if Triton unavailable or on CPU
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)

        return output


def run(*args):
    return ModelNew()(*args)
