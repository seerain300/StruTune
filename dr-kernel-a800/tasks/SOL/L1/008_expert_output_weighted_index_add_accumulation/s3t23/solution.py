import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_atomic_scalar(out_ptr, expert_ptr, idx_ptr, H: tl.constexpr, N: tl.constexpr):
    # One program per update i
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Destination row index (int32)
    idx = tl.load(idx_ptr + pid)

    # Iterate over hidden dimension and perform per-element atomic add
    for j in range(H):
        vj = tl.load(expert_ptr + pid * H + j)
        # out_ptr is a contiguous [M, H] float32 buffer; row idx starts at idx * H
        tl.atomic_add(out_ptr + idx * H + j, vj)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure all inputs are on the same CUDA device and contiguous
        device = final_hidden_states.device
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA"

        # Shapes
        batch_seq_len, hidden_size = final_hidden_states.shape
        num_selected_tokens, H = expert_outputs.shape
        assert H == hidden_size, "expert_outputs second dim must match hidden_size"
        assert token_indices.shape[0] == num_selected_tokens, "token_indices length must match num_selected_tokens"

        # Prepare fp32 accumulation buffer from the bfloat16 input (clone to match initial state)
        out_fp32 = final_hidden_states.to(torch.float32).clone()

        # Cast expert_outputs to float32 (input was bfloat16)
        expert_outputs_fp32 = expert_outputs.to(torch.float32).contiguous()
        # token_indices should be int32 for Triton
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        # Launch Triton kernel: one program per update (row to add)
        N = num_selected_tokens
        grid = (N,)
        # num_warps tuned for small per-program work; adjust if needed
        scatter_add_rows_atomic_scalar[grid](out_fp32, expert_outputs_fp32, token_indices_i32, H=H, N=N, num_warps=4, num_stages=2)

        # Cast back to bfloat16 to match original output dtype
        result = out_fp32.to(torch.bfloat16)
        return result


def run(*args):
    return ModelNew()(*args)
