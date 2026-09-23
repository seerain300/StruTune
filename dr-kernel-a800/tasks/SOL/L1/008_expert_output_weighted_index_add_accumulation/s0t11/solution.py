import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_H': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_H': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_H': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_H': 64}, num_warps=2, num_stages=2),
    ],
    key=['N', 'H'],
)
@triton.jit
def scatter_add_f32_atomic_block_kernel(
    output_ptr,           # *f32, shape (M, H)
    expert_ptr,           # *f32, shape (N, H)
    token_indices_ptr,    # *i64, shape (N,)
    N: tl.int32,          # number of source rows (num_selected_tokens)
    H: tl.int32,          # hidden size
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 2D launch: axis 0 over N in blocks, axis 1 over H in blocks
    pid_n = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)            # [BLOCK_N]
    h_block = pid_h * BLOCK_H
    h_offsets = h_block + tl.arange(0, BLOCK_H)                    # [BLOCK_H]

    mask_n = n_offsets < N
    mask_h = h_offsets < H

    # Load token indices for these n rows (vectorized)
    idx_vec = tl.load(token_indices_ptr + n_offsets, mask=mask_n, other=0)  # [BLOCK_N], i64

    # Compute destination offsets for each (n, h)
    # dest = idx * H + h
    # We will broadcast idx_vec over h_offsets
    # Prepare 2D broadcasted offsets: shape (BLOCK_N, BLOCK_H)
    # Note: Triton supports broadcasting via [:, None] and [None, :]
    dest_offsets = idx_vec[:, None] * H + h_offsets[None, :]        # [BLOCK_N, BLOCK_H]

    # Load expert outputs for these rows and hidden blocks
    # expert_ptr has shape (N, H)
    # Address = expert_ptr + n_offsets[:, None] * H + h_offsets[None, :]
    expert_addrs = expert_ptr + n_offsets[:, None] * H + h_offsets[None, :]
    # Mask: valid rows and valid hidden columns
    mask_load = mask_n[:, None] & mask_h[None, :]
    vals = tl.load(expert_addrs, mask=mask_load, other=0.0)        # [BLOCK_N, BLOCK_H], f32

    # Atomic add into output (float32)
    out_addrs = output_ptr + dest_offsets
    tl.atomic_add(out_addrs, vals, mask=mask_load)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        """
        Triton-accelerated scatter-add along dim=0:
        output[token_indices[i]] += expert_outputs[i]
        Returns updated final_hidden_states with contributions added.
        """

        # Ensure devices and dtypes are consistent
        device = final_hidden_states.device
        assert final_hidden_states.is_cuda, "Input must be on CUDA device for Triton kernels"
        assert expert_outputs.is_cuda and token_indices.is_cuda, "All inputs must be on CUDA device"

        # Clone to avoid modifying the original buffer
        # Cast to float32 for robust atomic add; we'll cast back to bfloat16 at the end.
        output_f32 = final_hidden_states.to(torch.float32).clone()

        # Cast expert_outputs to float32 for atomic accumulation
        expert_f32 = expert_outputs.to(torch.float32)

        # Ensure token_indices is int64 for Triton
        # (PyTorch default is int64; if not, convert)
        if token_indices.dtype != torch.int64:
            token_indices = token_indices.to(torch.int64)

        # Shapes
        M, H = final_hidden_states.shape
        N = expert_outputs.shape[0]

        # Launch Triton kernel over 2D grid
        # Grid size: (ceil_div(N, BLOCK_N), ceil_div(H, BLOCK_H))
        grid = (triton.cdiv(N, 64), triton.cdiv(H, 128))  # initial grid; autotune will override configs

        scatter_add_f32_atomic_block_kernel[grid](
            output_f32, expert_f32, token_indices,
            N, H,
        )

        # Cast back to original dtype (bfloat16) to match the original API contract
        output = output_f32.to(final_hidden_states.dtype)

        return output


def run(*args):
    return ModelNew()(*args)
