import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_per_row_kernel(
    out_ptr,          # *bf16, shape (N, H), output buffer (clone of final_hidden_states)
    src_ptr,          # *bf16, shape (M, H), expert_outputs
    indices_ptr,      # *int32, shape (M,), token_indices
    N: tl.constexpr,  # number of rows in out (batch_seq_len)
    H: tl.constexpr,  # hidden size
):
    # Each program handles one source row 'pid'
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Load target row index for this source row
    row_idx = tl.load(indices_ptr + pid)
    if row_idx < 0 or row_idx >= N:
        return

    # Loop over hidden dimension and atomic add
    for j in range(0, H):
        # Source pointer offset: src_ptr + pid * H + j
        src_offset = pid * H + j
        src_val = tl.load(src_ptr + src_offset)
        # Destination pointer offset: out_ptr + row_idx * H + j
        dst_offset = row_idx * H + j
        # Atomic add into destination
        tl.atomic_add(out_ptr + dst_offset, src_val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure tensors are on CUDA and contiguous
        device = final_hidden_states.device
        if device.type != "cuda":
            # Triton requires CUDA; if not on CUDA, fallback to PyTorch (not ideal, but keep correctness)
            out = final_hidden_states.clone()
            out.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return out

        # Clone to match PyTorch reference behavior exactly
        out = final_hidden_states.clone()

        # Make sure inputs are contiguous
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.to(torch.int32).contiguous()

        # Dimensions
        N = final_hidden_states.shape[0]  # batch_seq_len
        M = expert_outputs.shape[0]       # num_selected_tokens
        H = final_hidden_states.shape[1]  # hidden_size

        # Launch Triton kernel: 1 program per source row
        grid = (N,)
        scatter_add_per_row_kernel[grid](
            out, expert_outputs, token_indices,
            N=N, H=H,
            num_warps=1,  # correctness first; modest warp count
            num_stages=1,
        )
        return out


def run(*args):
    return ModelNew()(*args)
