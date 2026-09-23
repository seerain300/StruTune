import torch
import triton
import triton.language as tl


@triton.jit
def full_row_kernel(
    out_ptr,          # *fp32 [M, H]
    idx_ptr,          # *i32 [N]
    src_ptr,          # *fp32 [N, H]
    N,                # int32
    H,                # int32
    T: tl.constexpr,        # number of updates per program
    BLOCK_H: tl.constexpr,  # must equal H for this kernel
):
    pid = tl.program_id(0)
    start = pid * T
    lanes = tl.arange(0, T)
    pos = start + lanes
    lane_mask = pos < N

    for k in range(T):
        pos_k = pos[k]
        mask_k = lane_mask[k]

        # Load index for this update
        idx = tl.load(idx_ptr + pos_k, mask=mask_k, other=0)  # i32

        # Base pointers for this row
        out_row_ptr = out_ptr + idx * H
        src_row_ptr = src_ptr + pos_k * H

        # Vector of H elements
        offs = tl.arange(0, BLOCK_H)
        h_mask = offs < H & mask_k

        # Load vector and atomic add (full row at once)
        v = tl.load(src_row_ptr + offs, mask=h_mask, other=0.0)
        tl.atomic_add(out_row_ptr + offs, v, mask=h_mask)


@triton.jit
def chunk_kernel(
    out_ptr,          # *fp32 [M, H]
    idx_ptr,          # *i32 [N]
    src_ptr,          # *fp32 [N, H]
    N,                # int32
    H,                # int32
    T: tl.constexpr,        # number of updates per program
    BLOCK_H: tl.constexpr,  # chunk size for hidden dim
):
    pid = tl.program_id(0)
    start = pid * T

    for k in range(T):
        pos = start + k
        mask_k = pos < N

        # Load index for this update
        idx = tl.load(idx_ptr + pos, mask=mask_k, other=0)  # i32

        out_row_ptr = out_ptr + idx * H
        src_row_ptr = src_ptr + pos * H

        # Process hidden dimension in chunks
        for h in range(0, H, BLOCK_H):
            offs = h + tl.arange(0, BLOCK_H)
            mask = (offs < H) & mask_k
            v = tl.load(src_row_ptr + offs, mask=mask, other=0.0)
            tl.atomic_add(out_row_ptr + offs, v, mask=mask)


def _triton_scatter_add_fused(out_fp32, expert_outputs_fp32, token_indices_i32, N, H):
    """
    out_fp32: [M, H] float32
    expert_outputs_fp32: [N, H] float32
    token_indices_i32: [N] int32
    Updates out_fp32 via scatter-add: out_fp32[token_indices[i]] += expert_outputs_fp32[i, :].
    """
    # Fast path for common hidden_size (<= 1024): full-row atomic add
    if H <= 1024:
        BLOCK_H = 1024
        T = 8  # process 8 updates per program
        grid = (triton.cdiv(N, T),)
        full_row_kernel[grid](
            out_fp32,
            token_indices_i32,
            expert_outputs_fp32,
            N,
            H,
            T=T,
            BLOCK_H=BLOCK_H,
            num_warps=8,
            num_stages=2,
        )
    else:
        # Chunked path for larger hidden sizes
        if H <= 2048:
            BLOCK_H = 512
            num_warps = 4
        else:
            BLOCK_H = 256
            num_warps = 2
        T = 4
        grid = (triton.cdiv(N, T),)
        chunk_kernel[grid](
            out_fp32,
            token_indices_i32,
            expert_outputs_fp32,
            N,
            H,
            T=T,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=2,
        )
    return out_fp32


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add:
          final_hidden_states[token_indices[i]] += expert_outputs[i, :]
        Returns updated final_hidden_states.
        """
        # Ensure all tensors are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Cast indices to int32 for Triton
        token_indices_i32 = token_indices.to(torch.int32)

        # Cast accumulation to float32 (bf16 atomics not supported)
        out_fp32 = final_hidden_states.to(torch.float32).clone()

        # Cast expert outputs to float32 for accumulation
        expert_outputs_fp32 = expert_outputs.to(torch.float32)

        N = expert_outputs_fp32.shape[0]
        H = expert_outputs_fp32.shape[1]

        # Launch Triton kernel
        out_fp32 = _triton_scatter_add_fused(out_fp32, expert_outputs_fp32, token_indices_i32, N, H)

        # Cast back to bfloat16 to match original API
        return out_fp32.to


def run(*args):
    return ModelNew()(*args)
