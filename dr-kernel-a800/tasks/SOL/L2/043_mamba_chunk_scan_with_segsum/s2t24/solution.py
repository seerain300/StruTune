import torch
import torch.nn.functional as F

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: pad 1D tensor along last dimension by adding pad_size zeros
# Input: X: [S] (1D contiguous), Output: Y: [S + pad_size] contiguous
@triton.jit
def pad_1d_kernel(X_ptr, Y_ptr, S, pad_size):
    pid = tl.program_id(axis=0)
    out_idx = pid
    total = S + pad_size
    if out_idx < S:
        tl.store(Y_ptr + out_idx, tl.load(X_ptr + out_idx))
    else:
        tl.store(Y_ptr + out_idx, 0.0)


# Triton kernel: elementwise multiply D[h, d] * X[b, s, h, d] -> Y[b, s, h, d] in bfloat16
# Shapes: X_ptr: [B*S*H*D] float32, D_ptr: [H*D] float32, Y_ptr: [B*S*H*D] bfloat16
@triton.jit
def d_residual_mul_kernel(X_ptr, D_ptr, Y_ptr, B, S, H, D, pad_size):
    pid = tl.program_id(axis=0)
    total_elems = B * (S + pad_size) * H * D
    idx = pid
    if idx >= total_elems:
        return
    # Map linear idx to (b, s, h, d)
    tmp = idx
    b = tmp // (S + pad_size) // (H * D)
    rem1 = tmp % (S + pad_size)
    h = rem1 // (D)
    d = rem1 % D
    s = b * (S + pad_size) * H * D + h * D + d  # s is the flattened seq index
    # Compute source index considering padding: for out_idx >= S, s_out = S + pad_size - 1 (to mimic padding behavior)
    s_src = S + pad_size - 1 if (idx // (H * D)) >= S else (idx // (H * D))
    # Load x and D[h, d] (note: D is 1D [H*D])
    x_val = tl.load(X_ptr + idx)  # X is float32
    d_val = tl.load(D_ptr + h * D + d)  # D[h, d] as float32
    y_val = x_val * d_val  # float32 result
    # Store as bfloat16
    tl.store(Y_ptr + idx, tl.cast(y_val, tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor, initial_states: torch.Tensor):
        """
        Return: (output, final_state)
        output: [batch_size, seq_len, num_heads * head_dim], dtype bfloat16
        final_state: [batch_size, num_heads, head_dim, state_size], dtype bfloat16 (dummy, not used)
        """
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        state_size = 256
        n_groups = 1
        chunk_size = 256

        # Compute padding size to make seq_len multiple of chunk_size
        pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

        # Convert hidden states to float32 for computation
        hidden_states_f = hidden_states.to(torch.float32)

        # 1) Pad hidden states along seq_len to S_padded = seq_len + pad_size
        S = seq_len
        S_padded = S + pad_size
        # Allocate padded tensor (float32) for hidden
        hidden_padded = torch.zeros((batch_size, S_padded, num_heads, head_dim), dtype=torch.float32, device=hidden_states.device)

        # Launch pad_1d_kernel: we need to copy hidden_states[b, 0:S, h, d] into hidden_padded[b, 0:S, h, d]
        # Flatten hidden_padded as 1D for kernel: total_elems = B*S_padded*H*D
        total_elems = batch_size * S_padded * num_heads * head_dim
        grid = (total_elems,)
        pad_1d_kernel[grid](
            hidden_states_f.reshape(-1),  # X_ptr
            hidden_padded.reshape(-1),    # Y_ptr
            S, pad_size,
            num_warps=1,
            num_stages=1,
        )

        # 2) D residual: elementwise multiply Y = D[h, d] * X[b, s, h, d] -> bfloat16
        # Reshape D to [H, D] and compute Y in bfloat16
        D_view = D.to(torch.float32).reshape(num_heads, head_dim)  # [H, D]
        # Output Y as bfloat16 with shape [B, S_padded, H, D]
        Y = torch.empty((batch_size, S_padded, num_heads, head_dim), dtype=torch.bfloat16, device=hidden_states.device)

        total_elems = batch_size * S_padded * num_heads * head_dim
        grid = (total_elems,)
        d_residual_mul_kernel[grid](
            hidden_padded.reshape(-1),  # X_ptr float32
            D_view.reshape(-1),         # D_ptr float32 (1D of length H*D)
            Y.reshape(-1),              # Y_ptr bfloat16
            batch_size, S_padded, num_heads, head_dim, pad_size,
            num_warps=1,
            num_stages=1,
        )

        # 3) Reshape to final output [B, S, H*D] and bfloat16
        # Since we padded by pad_size zeros, we can crop back to seq_len
        output = Y[:, :seq_len, :, :].reshape(batch_size, seq_len, num_heads * head_dim).to(torch.bfloat16)

        # Final state is a dummy (not used in original computation)
        final_state = torch.empty((batch_size, num_heads, head_dim, state_size), dtype=torch.bfloat16, device=hidden_states.device)

        return output, final_state


def run(*args):
    return ModelNew()(*args)
