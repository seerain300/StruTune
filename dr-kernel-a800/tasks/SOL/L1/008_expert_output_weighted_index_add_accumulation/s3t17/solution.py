import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,           # *fp32, pointer to out_fp32, shape [M, H] as 1D
    idx_ptr,           # *int32, token_indices, shape [N]
    src_ptr,           # *fp32, expert_outputs, shape [N, H] as 1D
    M: tl.constexpr,   # number of rows (batch_seq_len)
    H: tl.constexpr,   # hidden_size (columns)
    N: tl.constexpr,   # number of updates (num_selected_tokens)
):
    # One program per update i
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Load destination row index
    idx = tl.load(idx_ptr + pid)
    if idx < 0 or idx >= M:
        return

    # Iterate over columns and add src row to out row
    # We process the entire row in a simple loop; Triton supports scalar loops here.
    # Each program handles a single row, so no race conditions.
    for col in range(0, H):
        # out[row, col]
        out_off = idx * H + col
        # src[i, col]
        src_off = pid * H + col
        v_out = tl.load(out_ptr + out_off)
        v_src = tl.load(src_ptr + src_off)
        v_new = v_out + v_src
        tl.store(out_ptr + out_off, v_new)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        """
        Triton-only implementation of:
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices, expert_outputs)
        Returns output with dtype bfloat16, matching the original run.
        """
        # Ensure device and contiguity (the provided get_inputs already makes them CUDA and contiguous,
        # but we enforce it for safety).
        assert final_hidden_states.is_cuda, "final_hidden_states must be on CUDA device"
        assert expert_outputs.is_cuda, "expert_outputs must be on CUDA device"
        assert token_indices.is_cuda, "token_indices must be on CUDA device"
        assert final_hidden_states.is_contiguous(), "final_hidden_states must be contiguous"
        assert expert_outputs.is_contiguous(), "expert_outputs must be contiguous"
        assert token_indices.is_contiguous(), "token_indices must be contiguous"

        # Shapes
        M = final_hidden_states.shape[0]  # batch_seq_len
        H = final_hidden_states.shape[1]  # hidden_size
        # expert_outputs: [N, H], token_indices: [N]
        assert expert_outputs.shape[1] == H, "expert_outputs second dimension must equal hidden_size"
        N = expert_outputs.shape[0]

        # Prepare fp32 accumulation buffer: clone final_hidden_states as fp32
        out_fp32 = final_hidden_states.to(torch.float32).clone()

        # Convert expert_outputs to fp32
        src_fp32 = expert_outputs.to(torch.float32)

        # Convert token_indices to int32
        idx_i32 = token_indices.to(torch.int32)

        # Launch Triton kernel: one program per update (i)
        grid = (N,)
        scatter_add_rows_kernel[grid](
            out_fp32, idx_i32, src_fp32,
            M, H, N,
            num_warps=1, num_stages=1,
        )

        # Cast back to bfloat16 to match the original output dtype
        out_bf16 = out_fp32.to(torch.bfloat16)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
