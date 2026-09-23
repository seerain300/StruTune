import torch
import triton
import triton.language as tl

@triton.jit
def scatter_add_chunk_kernel(
    out_ptr,                # *fp32, shape [batch_seq_len, H]
    token_indices_ptr,      # *i32,  shape [N]
    expert_outputs_ptr,     # *fp32, shape [N, H]
    N,                      # number of updates
    H,                      # hidden_size
    T: tl.constexpr,        # updates per program
    BLOCK_H: tl.constexpr,  # chunk size over hidden dim
):
    # Each program handles a group of T updates
    pid = tl.program_id(axis=0)
    start = pid * T
    lanes = tl.arange(0, T)               # [T]
    pos = start + lanes                   # [T]
    mask_update = pos < N                 # [T], true for valid lanes

    # For each lane, get the destination index
    idx = tl.load(token_indices_ptr + pos, mask=mask_update, other=0)  # [T] int32

    # Iterate over hidden dimension in chunks of BLOCK_H
    for h in range(0, H, BLOCK_H):
        offs = h + tl.arange(0, BLOCK_H)  # [BLOCK_H]
        mask_h = offs < H
        # Build 2D mask [T, BLOCK_H]: only operate if update is valid and offs within H
        mask = mask_update[:, None] & mask_h[None, :]

        # Load v_chunk for each lane (vector of length BLOCK_H)
        v = tl.load(expert_outputs_ptr + pos[:, None] * H + offs[None, :],
                    mask=mask, other=0.0)  # [T, BLOCK_H] fp32

        # Compute output pointers for atomic adds: out[idx, offs]
        out_ptrs = out_ptr + idx[:, None] * H + offs[None, :]
        # Atomic add the chunk
        tl.atomic_add(out_ptrs, v, mask=mask)

# Helper to choose kernel launch parameters
def _choose_kernel_params(H: int):
    if H >= 1024:
        BLOCK_H = 1024
        num_warps = 8
    elif H >= 512:
        BLOCK_H = 512
        num_warps = 4
    else:
        BLOCK_H = 256
        num_warps = 2
    return BLOCK_H, num_warps

class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized scatter-add:
        out[token_indices[i]] += expert_outputs[i, :]
        Accumulate in float32 (bf16 atomics not supported), then cast back to bfloat16.
        """
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA device for Triton."

        # Ensure contiguity and dtypes
        out_fp32 = final_hidden_states.to(torch.float32).contiguous()
        expert_outputs_fp32 = expert_outputs.to(torch.float32).contiguous()
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        batch_seq_len = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        # N is number of updates: batch_seq_len * len(token_indices)
        N = batch_seq_len * token_indices_i32.numel()

        # Choose kernel params
        BLOCK_H, num_warps = _choose_kernel_params(H)
        # Process more updates per program to amortize overhead
        T = 8

        grid = (triton.cdiv(N, T),)

        scatter_add_chunk_kernel[grid](
            out_fp32, token_indices_i32, expert_outputs_fp32,
            N, H,
            T=T, BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=2,
        )

        # Cast back to bfloat16 to match original API
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
