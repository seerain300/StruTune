import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per update; atomically add expert_outputs[i, :] to out_fp32[token_indices[i], :]
if TRITON_AVAILABLE:
    @triton.jit
    def scatter_add_rows_atomic_kernel(
        out_ptr,        # *float32, [N_rows, H]
        A_ptr,          # *float32, [N_updates, H]
        indices_ptr,    # *int32,  [N_updates]
        N_updates: tl.int32,  # total number of updates
        N_rows: tl.int32,     # total number of rows (batch_seq_len)
        H: tl.int32,          # hidden size
        num_warps: tl.constexpr,
    ):
        # Program id corresponds to the update index
        i = tl.program_id(0)
        # Optional guard (grid may be larger than N_updates)
        if i >= N_updates:
            return

        # Load destination row index and the vector to add
        idx = tl.load(indices_ptr + i)  # int32

        # Compute base pointers for this row and vector
        out_row_ptr = out_ptr + idx * H
        A_row_ptr = A_ptr + i * H

        # Load H elements of the vector to add
        offsets = tl.arange(0, H)
        # For safety, mask is always true but keeps Triton happy for edge cases
        mask = offsets < H
        A_vec = tl.load(A_row_ptr + offsets, mask=mask, other=0.0)

        # Atomically add to the destination row
        tl.atomic_add(out_row_ptr + offsets, A_vec, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of: output.index_add_(dim=0, token_indices, expert_outputs)
        Accumulates in float32 using atomic adds, then returns bfloat16 result.
        """
        assert final_hidden_states.dim() == 2, "final_hidden_states must be 2D [batch_seq_len, hidden_size]"
        assert expert_outputs.dim() == 2, "expert_outputs must be 2D [num_selected_tokens, hidden_size]"
        assert token_indices.dim() == 1, "token_indices must be 1D [num_selected_tokens]"

        batch_seq_len, hidden_size = final_hidden_states.shape
        num_selected_tokens, out_hidden_size = expert_outputs.shape
        assert token_indices.numel() == num_selected_tokens, "token_indices length must match num_selected_tokens"
        # Optional bounds check to avoid undefined behavior (even though get_inputs ensures valid indices)
        assert token_indices.min().item() >= 0 and token_indices.max().item() < batch_seq_len, "token_indices out of bounds"

        # If Triton not available, fall back to PyTorch for correctness
        if not TRITON_AVAILABLE or final_hidden_states.device.type != 'cuda' or expert_outputs.device.type != 'cuda':
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices, expert_outputs)
            return output

        # Ensure inputs are on the same device and contiguous
        device = final_hidden_states.device
        assert expert_outputs.device == device and token_indices.device == device, "All inputs must be on the same device"
        if not final_hidden_states.is_contiguous():
            final_hidden_states = final_hidden_states.contiguous()
        if not expert_outputs.is_contiguous():
            expert_outputs = expert_outputs.contiguous()
        if token_indices.dtype != torch.int64:
            # Keep as int64 for safety; Triton kernel loads as int32 and compares safely
            token_indices = token_indices.to(torch.int64)

        # Upcast to float32 for accumulation (bfloat16 does not support atomic_add)
        out_fp32 = final_hidden_states.to(torch.float32).clone()
        A_fp32 = expert_outputs.to(torch.float32)
        indices_i32 = token_indices.to(torch.int32)

        # Launch Triton kernel: one program per update
        N_updates = num_selected_tokens
        N_rows = batch_seq_len
        H = hidden_size

        # Choose num_warps based on hidden size
        if H >= 1024:
            num_warps = 8
        elif H >= 512:
            num_warps = 4
        else:
            num_warps = 2

        grid = (N_updates,)  # one program per update

        scatter_add_rows_atomic_kernel[grid](
            out_fp32, A_fp32, indices_i32,
            N_updates, N_rows, H,
            num_warps=num_warps,
        )

        # Cast back to bfloat16 to match original dtype
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
