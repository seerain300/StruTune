import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_atomic_rows_kernel(
    out_ptr,          # *bf16, pointer to out tensor [N, H]
    src_ptr,          # *bf16, pointer to src tensor [M, H]
    indices_ptr,      # *int32, pointer to token indices [M]
    N,                # int32, number of rows in out (batch_seq_len)
    H,                # int32, hidden size
    M,                # int32, number of expert outputs
):
    pid = tl.program_id(0)
    # One program per row i in [0, M)
    i = pid  # grid size is M
    if i >= M:
        return

    # Load destination row index (int32)
    row_idx = tl.load(indices_ptr + i)

    # Iterate over hidden dimension and perform atomic adds
    # This avoids vectorized pointer arithmetic to stay compatible
    for j in range(0, H):
        src_offset = i * H + j
        out_offset = row_idx * H + j
        val = tl.load(src_ptr + src_offset)
        tl.atomic_add(out_ptr + out_offset, val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-only forward that performs:
          out = final_hidden_states.clone()
          out[token_indices[i]] += expert_outputs[i] for all i
        All compute happens inside Triton kernels.
        """
        # Ensure tensors are contiguous
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Clone to match reference behavior exactly
        out = final_hidden_states.clone()

        # Triton requires int32 indices for pointer arithmetic
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Launch Triton kernel: one program per source row i
        M = expert_outputs.shape[0]  # number of expert outputs
        grid = (M,)
        scatter_add_atomic_rows_kernel[grid](
            out, expert_outputs, token_indices,
            out.shape[0], out.shape[1], M,
            num_warps=1,  # keep conservative to avoid register pressure
        )

        return out


def run(*args):
    return ModelNew()(*args)
