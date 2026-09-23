import torch
import triton
import triton.language as tl


@triton.jit
def full_row_add_kernel(
    out_ptr,                # *float32, [M, H]
    token_indices_ptr,      # *int32,   [N]
    expert_outputs_ptr,     # *float32, [N, H]
    N: tl.constexpr,        # number of updates
    H: tl.constexpr,        # hidden size
    T: tl.constexpr,        # updates per program
    BLOCK_H: tl.constexpr,  # set to H for full-row
):
    pid = tl.program_id(axis=0)
    # Base update index this program will handle
    base = pid * T
    # Process up to T updates, masking for tail
    for k in range(T):
        pos = base + k
        valid = pos < N
        # Load index
        idx = tl.load(token_indices_ptr + pos, mask=valid, other=0)
        # Load the entire expert row into a vector
        offs = tl.arange(0, BLOCK_H)
        v = tl.load(expert_outputs_ptr + pos * H + offs, mask=valid, other=0.0)
        # Atomic add into output row
        tl.atomic_add(out_ptr + idx * H + offs, v, mask=valid)


@triton.jit
def chunked_add_kernel(
    out_ptr,                # *float32, [M, H]
    token_indices_ptr,      # *int32,   [N]
    expert_outputs_ptr,     # *float32, [N, H]
    N: tl.constexpr,        # number of updates
    H: tl.constexpr,        # hidden size
    T: tl.constexpr,        # updates per program
    BLOCK_H: tl.constexpr,  # e.g., 1024/512/256
):
    pid = tl.program_id(axis=0)
    base = pid * T
    for k in range(T):
        pos = base + k
        valid = pos < N
        idx = tl.load(token_indices_ptr + pos, mask=valid, other=0)
        # Iterate across hidden dimension in BLOCK_H chunks
        h = 0
        while h < H:
            offs = h + tl.arange(0, BLOCK_H)
            mask = valid & (offs < H)
            v = tl.load(expert_outputs_ptr + pos * H + offs, mask=mask, other=0.0)
            tl.atomic_add(out_ptr + idx * H + offs, v, mask=mask)
            h += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure device and contiguity; accumulate in float32
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA"
        M = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]

        # Clone and cast to float32 for accumulation (bfloat16 atomic add not supported)
        out_fp32 = final_hidden_states.to(torch.float32).clone()
        expert_outputs_fp32 = expert_outputs.to(torch.float32)
        token_indices_i32 = token_indices.to(torch.int32)

        # Choose kernel and launch parameters based on H
        # Heuristics: full-row kernel when H <= 1024, chunked otherwise
        if H <= 1024:
            BLOCK_H = 1024
            T = 8  # process up to 8 updates per program
            num_warps = 8
            grid = (triton.cdiv(N, T),)
            full_row_add_kernel[grid](
                out_fp32, token_indices_i32, expert_outputs_fp32, N, H, T, BLOCK_H,
                num_warps=num_warps, num_stages=2,
            )
        else:
            # Chunked kernel: choose BLOCK_H and T based on H
            # Use larger BLOCK_H for better throughput
            if H >= 2048:
                BLOCK_H = 1024
                num_warps = 8
            elif H >= 1024:
                BLOCK_H = 512
                num_warps = 4
            else:
                BLOCK_H = 256
                num_warps = 2
            T = 8
            grid = (triton.cdiv(N, T),)
            chunked_add_kernel[grid](
                out_fp32, token_indices_i32, expert_outputs_fp32, N, H, T, BLOCK_H,
                num_warps=num_warps, num_stages=2,
            )

        # Cast back to bfloat16 to match original API
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
