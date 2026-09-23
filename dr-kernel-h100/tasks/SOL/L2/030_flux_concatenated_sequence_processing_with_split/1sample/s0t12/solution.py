import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_sequences_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, I, H,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_i, stride_h_h,
    stride_o_b, stride_o_t, stride_o_h,
    BLOCK_L: tl.constexpr,
):
    b = tl.program_id(0)
    # Compute base pointers for this batch
    # encoder: [B, T, H], hidden: [B, I, H], out: [B, T+I, H]
    if b >= B:
        return
    # Loop over sequence positions
    for l in range(0, T + I, BLOCK_L):
        idx = l + tl.arange(0, BLOCK_L)
        mask = idx < (T + I)
        # Select source based on l < T
        is_encoder = idx < T
        # Compute source pointers and load
        src_rows = tl.where(is_encoder, idx, idx - T)  # rows in hidden are l - T
        # Pointer arithmetic for encoder loads
        e_ptrs = encoder_ptr + b * stride_e_b + src_rows * stride_e_t + tl.arange(0, H) * stride_e_h
        # Pointer arithmetic for hidden loads
        h_ptrs = hidden_ptr + b * stride_h_b + (idx - T) * stride_h_i + tl.arange(0, H) * stride_h_h
        # Load with mask
        vals = tl.load(
            tl.where(is_encoder, e_ptrs, h_ptrs),
            mask=mask & (tl.arange(0, H) < H),
            other=0.0,
        )
        # Store to out[b, idx, :]
        out_ptrs = out_ptr + b * stride_o_b + idx[:, None] * stride_o_t + tl.arange(0, H) * stride_o_h
        tl.store(out_ptrs, vals, mask=mask[:, None])


@triton.jit
def _triton_matmul_kernel(
    A_ptr, W_ptr, C_ptr,
    M, K, N,  # here N == H, but keep as general
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over output tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Loop over K
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        # Load W tile as [BLOCK_K, BLOCK_N] (we want W[n, k])
        w_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        # Accumulate
        acc += tl.dot(a, w)
    # Store results to C
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _split_streams_kernel(
    C_ptr, out_encoder_ptr, out_hidden_ptr,
    B, T, I, H,
    stride_c_m, stride_c_n,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_i, stride_h_h,
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)
    if b >= B:
        return
    total = T + I
    # First copy encoder part: rows [0, T)
    for l in range(0, T):
        m = b * total + l
        # Loop over hidden dim H in chunks
        for h0 in range(0, H, BLOCK_H):
            h_idx = h0 + tl.arange(0, BLOCK_H)
            mask = h_idx < H
            # Load C[m, h_idx]
            c_ptrs = C_ptr + m * stride_c_m + h_idx * stride_c_n
            vals = tl.load(c_ptrs, mask=mask, other=0.0)
            # Store to encoder output [b, l, :]
            out_ptrs = out_encoder_ptr + b * stride_e_b + l * stride_e_t + h_idx * stride_e_h
            tl.store(out_ptrs, vals, mask=mask)
    # Then copy hidden part: rows [T, T+I)
    for l in range(T, T + I):
        m = b * total + l
        for h0 in range(0, H, BLOCK_H):
            h_idx = h0 + tl.arange(0, BLOCK_H)
            mask = h_idx < H
            c_ptrs = C_ptr + m * stride_c_m + h_idx * stride_c_n
            vals = tl.load(c_ptrs, mask=mask, other=0.0)
            out_ptrs = out_hidden_ptr + b * stride_h_b + (l - T) * stride_h_i + h_idx * stride_h_h
            tl.store(out_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        device = hidden_states.device

        # Allocate concatenated tensor [B, T+I, H]
        out_cat = torch.empty((B, T + I, H), dtype=hidden_states.dtype, device=device)

        # Launch concatenation kernel: one program per batch
        _concatenate_sequences_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            *encoder_hidden_states.stride(),
            *hidden_states.stride(),
            *out_cat.stride(),
            BLOCK_L=256,
            num_warps=1,
            num_stages=1,
        )

        # Prepare A: out_cat as [M, K], M = B*(T+I), K = H
        A = out_cat
        M = B * (T + I)
        K = H

        # Prepare W: process_weight is [H, H], we right-multiply by W^T -> use W as [K, N] where N=H
        # We can pass W directly [H, H]; in kernel we load W[n, k] as [BLOCK_K, BLOCK_N] by flipping indices.
        W = process_weight.contiguous()  # ensure contiguous
        N = H  # output columns == hidden_dim

        # Allocate C: [M, H]
        C = torch.empty((M, N), dtype=torch.float32, device=device)  # compute in fp32 for stability

        # Choose tiling parameters
        # For typical H up to a few thousand and M up to tens of thousands, these tiles are reasonable.
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        grid = (
            triton.cdiv(M, BLOCK_M),
            triton.cdiv(N, BLOCK_N),
        )

        # Launch Triton matmul: compute C = A @ W
        _triton_matmul_kernel[grid](
            A, W, C,
            M, K, N,
            A.stride(0), A.stride(1),
            W.stride(0), W.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Allocate outputs [B, T, H] and [B, I, H]
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=device)

        # Launch split kernel: one program per batch
        _split_streams_kernel[(B,)](
            C, processed_encoder, processed_hidden,
            B, T, I, H,
            *C.stride(),
            *processed_encoder.stride(), *processed_hidden.stride(),
            BLOCK_H=256,
            num_warps=1,
            num_stages=1,
        )

        # If original dtype was not fp32, cast back (the original model uses fp32 by default)
        if hidden_states.dtype != torch.float32:
            processed_encoder = processed_encoder.to(hidden_states.dtype)
            processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
