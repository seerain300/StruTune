import torch

# Triton import guarded to avoid runtime errors if Triton is unavailable
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Robust Triton kernel: one program per expert output row, scalar loop over hidden dimension, per-element atomic add.
if TRITON_AVAILABLE:
    @triton.jit
    def scatter_add_rows_atomic_64_kernel(
        out_ptr,   # *bf16, shape [N, H], contiguous
        src_ptr,   # *bf16, shape [M, H], contiguous
        idx_ptr,   # *int64, shape [M], contiguous
        N,         # int32: number of rows in out (batch_seq_len)
        H,         # int32: hidden_size
        M          # int32: number of expert outputs
    ):
        i = tl.program_id(0)  # which expert output row
        # Guard if grid > M (not strictly necessary if grid==M)
        if i >= M:
            return

        # Destination row index as int64 for 64-bit addressing
        row_idx = tl.load(idx_ptr + i)  # int64

        # Base offsets in int64
        row_base = row_idx * H  # int64
        i_base = i * H          # int64

        # Scalar loop over hidden dimension, atomic add per element
        for j in range(0, H):
            dst_offset = row_base + j
            src_offset = i_base + j
            val = tl.load(src_ptr + src_offset)  # bf16
            tl.atomic_add(out_ptr + dst_offset, val)


def _run_triton_scatter_add(final_hidden_states: torch.Tensor,
                            expert_outputs: torch.Tensor,
                            token_indices: torch.Tensor) -> torch.Tensor:
    """
    Triton-based implementation of out.index_add_(dim=0, token_indices, expert_outputs).
    - final_hidden_states: (N, H), bfloat16, CUDA, contiguous
    - expert_outputs: (M, H), bfloat16, CUDA, contiguous
    - token_indices: (M,), int64, CUDA, contiguous
    Returns out (clone of final_hidden_states) with accumulations.
    """
    assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda
    assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16
    assert token_indices.dtype == torch.long, "token_indices must be int64 (torch.long)"

    # Clone to match PyTorch reference behavior exactly
    out = final_hidden_states.clone().contiguous()

    # Ensure inputs are contiguous
    expert_outputs = expert_outputs.contiguous()
    token_indices = token_indices.contiguous()

    N = out.shape[0]
    H = out.shape[1]
    M = expert_outputs.shape[0]

    # Triton grid: one program per expert output row
    grid = (M,)

    scatter_add_rows_atomic_64_kernel[grid](
        out, expert_outputs, token_indices,
        N, H, M,
        num_warps=4,   # modest warps for stability
        num_stages=2,  # typical for memory-bound kernels
    )
    return out


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        # If Triton or CUDA not available, fall back to PyTorch for correctness
        if (not TRITON_AVAILABLE) or (final_hidden_states.device.type != "cuda"):
            out = final_hidden_states.clone()
            out.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return out

        # Ensure dtypes and devices
        if final_hidden_states.dtype != torch.bfloat16:
            final_hidden_states = final_hidden_states.to(torch.bfloat16)
        if expert_outputs.dtype != torch.bfloat16:
            expert_outputs = expert_outputs.to(torch.bfloat16)
        if token_indices.dtype != torch.long:
            token_indices = token_indices.to(torch.long)

        return _run_triton_scatter_add(final_hidden_states, expert_outputs, token_indices)


def run(*args):
    return ModelNew()(*args)
