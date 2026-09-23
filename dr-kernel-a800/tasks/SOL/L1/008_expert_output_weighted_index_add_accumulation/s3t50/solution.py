import torch
import triton
import triton.language as tl


@triton.jit
def full_row_kernel(
    out_ptr,           # *float32 [B, H]
    token_indices_ptr, # *int32 [N]
    expert_ptr,        # *float32 [N, H]
    N,                 # int32
    B,                 # int32
    H,                 # int32
    T: tl.constexpr,   # number of updates per program
    BLOCK_H: tl.constexpr,  # must equal H here
    NUM_UPDATES: tl.constexpr,  # grid * T should be >= N
):
    pid = tl.program_id(axis=0)
    # Each program handles up to T updates
    start = pid * T
    # Loop over T updates (masked)
    for k in range(T):
        pos = start + k
        # Mask for valid updates
        if pos < N:
            # Load index
            idx = tl.load(token_indices_ptr + pos)
            # Compute row offset in out
            out_row_ptr = out_ptr + idx * H
            # Load entire row of expert output as a vector
            row_vec = tl.load(expert_ptr + pos * H + tl.arange(0, BLOCK_H))
            # Atomic add the vector into output row
            tl.atomic_add(out_row_ptr + tl.arange(0, BLOCK_H), row_vec)


@triton.jit
def chunk_kernel(
    out_ptr,                   # *float32 [B, H]
    token_indices_ptr,         # *int32 [N]
    expert_ptr,                # *float32 [N, H]
    N,                         # int32
    H,                         # int32
    T: tl.constexpr,           # number of updates per program
    BLOCK_H: tl.constexpr,     # e.g., 1024, 512, 256
):
    pid = tl.program_id(axis=0)
    start = pid * T
    # Process up to T updates in this program
    for k in range(T):
        pos = start + k
        if pos < N:
            # Load index
            idx = tl.load(token_indices_ptr + pos)
            # Pointer to output row base
            out_row_ptr = out_ptr + idx * H
            # Iterate over hidden dimension in BLOCK_H chunks
            for h in range(0, H, BLOCK_H):
                offs = h + tl.arange(0, BLOCK_H)
                # Load chunk of expert output row
                v = tl.load(expert_ptr + pos * H + offs)
                # Atomic add chunk into output row
                tl.atomic_add(out_row_ptr + offs, v)


def _choose_block_h(H: int):
    # Prefer full row if it's a convenient block size; else pick a large chunk.
    if H <= 1024:
        return 1024
    elif H <= 2048:
        return 512
    else:
        return 256


def _choose_warps(block_h: int):
    # Simple heuristic for num_warps
    if block_h >= 1024:
        return 8
    elif block_h >= 512:
        return 4
    else:
        return 2


def _choose_t(N: int):
    # Process 4 updates per program; last program may have fewer
    return 4


def triton_scatter_add_rowwise(final_hidden_states: torch.Tensor,
                               expert_outputs: torch.Tensor,
                               token_indices: torch.Tensor) -> torch.Tensor:
    """
    Triton-optimized scatter-add along rows:
    out[token_indices[i]] += expert_outputs[i, :]
    Accumulates in float32, returns bfloat16.
    """
    assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
        "All tensors must be on CUDA device for Triton kernels."

    # Ensure contiguous and dtype conversions
    B, H = final_hidden_states.shape
    N = expert_outputs.shape[0]
    # Accumulate in float32 for atomic_add support
    out_fp32 = final_hidden_states.clone().to(torch.float32)
    expert_fp32 = expert_outputs.to(torch.float32)
    token_i32 = token_indices.to(torch.int32)

    # Choose strategy
    block_h = _choose_block_h(H)
    num_warps = _choose_warps(block_h)
    T = _choose_t(N)

    # If we can handle full row without chunking
    if block_h >= H:
        grid = (triton.cdiv(N, T),)
        full_row_kernel[grid](
            out_fp32,
            token_i32,
            expert_fp32,
            N,
            B,
            H,
            T=T,
            BLOCK_H=block_h,
            num_warps=num_warps,
            num_stages=2,
        )
    else:
        # Chunked kernel for arbitrary H
        grid = (triton.cdiv(N, T),)
        chunk_kernel[grid](
            out_fp32,
            token_i32,
            expert_fp32,
            N,
            H,
            T=T,
            BLOCK_H=block_h,
            num_warps=num_warps,
            num_stages=2,
        )

    # Cast back to bfloat16 to match original API
    return out_fp32.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        # Triton-based scatter-add. Ensure device is CUDA.
        if not final_hidden_states.is_cuda:
            final_hidden_states = final_hidden_states.to('cuda')
        if not expert_outputs.is_cuda:
            expert_outputs = expert_outputs.to('cuda')
        if not token_indices.is_cuda:
            token_indices = token_indices.to('cuda')
        return triton_scatter_add_rowwise(final_hidden_states, expert_outputs, token_indices)


def run(*args):
    return ModelNew()(*args)
