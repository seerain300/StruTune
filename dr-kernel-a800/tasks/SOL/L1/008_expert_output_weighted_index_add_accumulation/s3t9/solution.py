import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_atomic_kernel(
    out_ptr,     # *fp32, shape [batch_seq_len, hidden_size]
    A_ptr,       # *fp32, shape [num_selected_tokens, hidden_size]
    idx_ptr,     # *int32, shape [num_selected_tokens]
    N,           # int32: number of updates (rows in A)
    H: tl.constexpr,  # hidden_size (compile-time for loop)
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Load the target row index for this update
    idx = tl.load(idx_ptr + pid)  # int32

    # Compute base offsets
    base_out = idx * H
    base_A = pid * H

    # Atomic add each element of A[pid, :] into out[base_out + j]
    # We use a simple loop to avoid dynamic vector complexity.
    for j in range(0, H):
        vj = tl.load(A_ptr + base_A + j)  # fp32 scalar
        tl.atomic_add(out_ptr + base_out + j, vj)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        # Ensure CUDA tensors and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "ModelNew.forward requires CUDA tensors"
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Dimensions
        batch_seq_len, hidden_size = final_hidden_states.shape
        num_selected_tokens = expert_outputs.shape[0]

        # Prepare fp32 accumulation buffer initialized to final_hidden_states (clone)
        out_fp32 = final_hidden_states.to(torch.float32).clone()

        # Prepare inputs for Triton: A as fp32, idx as int32
        A_fp32 = expert_outputs.to(torch.float32)
        idx_i32 = token_indices.to(torch.int32)

        # Launch one program per update (N)
        grid = (num_selected_tokens,)
        scatter_add_rows_atomic_kernel[grid](
            out_fp32, A_fp32, idx_i32,
            num_selected_tokens,
            H=hidden_size,
            num_warps=1,
        )

        # Cast back to bfloat16 to match original output dtype
        output = out_fp32.to(torch.bfloat16)
        return output


def run(*args):
    return ModelNew()(*args)
