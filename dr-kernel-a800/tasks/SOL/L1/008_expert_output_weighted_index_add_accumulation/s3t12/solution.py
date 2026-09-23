import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_add_rows_fp32(
    out_ptr,      # *fp32, shape [batch_seq_len, H]
    A_ptr,        # *fp32, shape [N, H]
    indices_ptr,  # *int32, shape [N]
    H: tl.constexpr,  # hidden size
    N: tl.constexpr,  # num_selected_tokens
    BLOCK: tl.constexpr = 1024,
):
    # One program per update i
    i = tl.program_id(0)
    # Safety: if i >= N (shouldn't happen with correct grid), exit
    if i >= N:
        return

    # Load destination row index for this update
    idx = tl.load(indices_ptr + i)

    # Vector of column offsets for a chunk of the hidden dimension
    offs = tl.arange(0, BLOCK)
    mask = offs < H

    # Load the contribution vector from A[i, :]
    # A is assumed to be contiguous in row-major: row i has base offset i*H
    # We load a masked vector of length H (BLOCK acts as upper bound)
    a = tl.load(A_ptr + i * H + offs, mask=mask, other=0.0)

    # Atomically add this vector to out[idx, :]
    # out is fp32 and contiguous: row base offset is idx*H
    tl.atomic_add(out_ptr + idx * H + offs, a, mask=mask)


@triton.jit
def _min_of_vector_fp32(out_ptr, x_ptr, L, BLOCK: tl.constexpr = 1024):
    # Compute minimum of vector x_ptr[L] and write to out_ptr[0] as fp32
    # Running min initialized to +inf
    running_min = tl.full((1,), float('inf'), tl.float32)
    # Iterate over chunks
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < L
        vals = tl.load(x_ptr + offs, mask=mask, other=float('inf'))
        chunk_min = tl.min(vals, axis=0)
        running_min = tl.minimum(running_min, chunk_min)
    tl.store(out_ptr, running_min)


@triton.jit
def _max_of_vector_fp32(out_ptr, x_ptr, L, BLOCK: tl.constexpr = 1024):
    # Compute maximum of vector x_ptr[L] and write to out_ptr[0] as fp32
    running_max = tl.full((1,), -float('inf'), tl.float32)
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < L
        vals = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))
        chunk_max = tl.max(vals, axis=0)
        running_max = tl.maximum(running_max, chunk_max)
    tl.store(out_ptr, running_max)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized version of:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)

        All computation is done in Triton kernels. We accumulate in fp32 and cast back to bfloat16.
        """
        assert final_hidden_states.dim() == 2, "final_hidden_states must be [batch_seq_len, hidden_size]"
        assert expert_outputs.dim() == 2, "expert_outputs must be [num_selected_tokens, hidden_size]"
        assert token_indices.dim() == 1, "token_indices must be 1D [num_selected_tokens]"

        device = final_hidden_states.device
        assert device.type == "cuda", "Triton kernels require CUDA device"
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Inputs must be on CUDA"

        # Shapes
        batch_seq_len, H = final_hidden_states.shape
        N, H_exp = expert_outputs.shape
        assert H_exp == H, "hidden_size must match between final_hidden_states and expert_outputs"
        assert token_indices.numel() == N, "num_selected_tokens must equal token_indices length"

        # Ensure contiguity and fp32 for accumulation
        out_fp32 = final_hidden_states.float().clone().contiguous()  # fp32 buffer for atomic accumulation
        A_fp32 = expert_outputs.float().contiguous()                 # cast expert_outputs to fp32
        idx_i32 = token_indices.int().contiguous()                  # cast token_indices to int32

        # Launch scatter-add kernel: one program per update
        grid = (N,)
        _scatter_add_rows_fp32[grid](
            out_fp32,
            A_fp32,
            idx_i32,
            H=H,
            N=N,
            BLOCK=1024,
            num_warps=4,
        )

        # Cast back to bfloat16 to match original dtype
        out_bf16 = out_fp32.to(torch.bfloat16)

        # Optional: Triton reductions to replace any host-side .min()/.max()
        # Example: compute min and max of final_hidden_states' values (for validation/demo)
        # Note: These calls can be removed or used for internal checks. The primary output is out_bf16.
        # min_val = torch.empty(1, dtype=torch.float32, device=device)
        # _min_of_vector_fp32[(1,)](min_val, final_hidden_states.float().contiguous().view(-1), final_hidden_states.numel())
        # max_val = torch.empty(1, dtype=torch.float32, device=device)
        # _max_of_vector_fp32[(1,)](max_val, final_hidden_states.float().contiguous().view(-1), final_hidden_states.numel())

        # No need to return min_val/max_val; the main result is out_bf16
        return out_bf16


def run(*args):
    return ModelNew()(*args)
