import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_atomic_fp32(
    out_ptr,        # *fp32, shape [M, H], output buffer initialized to clone of final_hidden_states
    A_ptr,          # *fp32, shape [N, H], expert outputs
    idx_ptr,        # *int32, shape [N], token indices
    N,              # int32, number of updates
    H: tl.constexpr # hidden size (compile-time constant for the kernel)
):
    # One program per update i
    i = tl.program_id(0)
    if i >= N:
        return

    # Load index and vector
    idx = tl.load(idx_ptr + i)  # int32
    offs = tl.arange(0, H)
    # Atomic add the H-length vector from A[i, :] into out[idx, :]
    v = tl.load(A_ptr + i * H + offs)
    tl.atomic_add(out_ptr + idx * H + offs, v)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Perform output = final_hidden_states.clone() and then output.index_add_(dim=0, token_indices, expert_outputs)
        using a Triton kernel with fp32 atomic adds, then cast back to bfloat16 to match original output dtype.
        """
        assert final_hidden_states.dim() == 2, "final_hidden_states must be 2D: [batch_seq_len, hidden_size]"
        assert expert_outputs.dim() == 2, "expert_outputs must be 2D: [num_selected_tokens, hidden_size]"
        assert token_indices.dim() == 1, "token_indices must be 1D: [num_selected_tokens]"
        batch_seq_len, H = final_hidden_states.shape
        N = expert_outputs.shape[0]
        assert token_indices.numel() == N, "token_indices length must equal number of expert outputs."

        # Ensure tensors are on the same CUDA device and contiguous
        device = final_hidden_states.device
        assert device.type == "cuda", "Triton kernels require CUDA tensors"
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Upcast to fp32 and initialize output from clone of final_hidden_states
        out_fp32 = final_hidden_states.to(torch.float32).clone()
        A_fp32 = expert_outputs.to(torch.float32)
        idx_i32 = token_indices.to(torch.int32)

        # Launch one program per update
        grid = (N,)
        scatter_add_rows_atomic_fp32[grid](
            out_fp32, A_fp32, idx_i32,
            N,
            H=H,  # hidden size as constexpr
            num_warps=4,  # adjust if needed
        )

        # Cast back to bfloat16 to match original output dtype
        output = out_fp32.to(torch.bfloat16)
        return output


def run(*args):
    return ModelNew()(*args)
