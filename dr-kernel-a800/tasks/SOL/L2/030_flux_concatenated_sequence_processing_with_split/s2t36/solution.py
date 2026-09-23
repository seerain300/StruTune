import torch
import triton
import triton.language as tl


@triton.jit
def concat_sequences_kernel(
    encoder_ptr,   # *ptr to [B, T, H]
    hidden_ptr,    # *ptr to [B, I, H]
    out_ptr,       # *ptr to [B, L, H], L = T + I
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    H: tl.constexpr,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_i, stride_h_h,
    stride_o_b, stride_o_m, stride_o_h,
    BLOCK_H: tl.constexpr,
):
    # program ids: b over batch, m over L sequence, h tile over H
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    # Determine region: encoder or hidden
    is_encoder = pid_m < T

    if is_encoder:
        # Load from encoder_hidden_states[b, m, h]
        e_ptrs = encoder_ptr + pid_b * stride_e_b + pid_m * stride_e_t + offs_h * stride_e_h
        o_ptrs = out_ptr + pid_b * stride_o_b + pid_m * stride_o_m + offs_h * stride_o_h
    else:
        # Load from hidden_states[b, m - T, h]
        i_pos = pid_m - T
        h_ptrs = hidden_ptr + pid_b * stride_h_b + i_pos * stride_h_i + offs_h * stride_h_h
        o_ptrs = out_ptr + pid_b * stride_o_b + pid_m * stride_o_m + offs_h * stride_o_h

    vals = tl.load(h_ptrs, mask=mask_h, other=0.0)
    tl.store(o_ptrs, vals, mask=mask_h)


@triton.jit
def bmm_kernel(
    A_ptr,       # *ptr to [B, M, K] where M = L
    B_ptr,       # *ptr to [K, N] where K = H, N = H (W_T)
    C_ptr,       # *ptr to [B, M, N] where N = H
    B: tl.constexpr,   # batch size (unused but can be used for grouping)
    M: tl.constexpr,   # sequence length (L)
    N: tl.constexpr,   # output hidden dim (H)
    K: tl.constexpr,   # reduction dim (H)
    stride_a_b, stride_a_m, stride_a_k,
    stride_b_k, stride_b_n,
    stride_c_b, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # grid over (batch, m tiles, n tiles)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # A tiles: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + pid_b * stride_a_b + offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B tiles: [BLOCK_K, BLOCK_N] (W_T[k, n])
        b_ptrs = B_ptr + offs_k[:, None] * stride_b_k + offs_n[None, :] * stride_b_n
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # accumulate
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # store result
    c_ptrs = C_ptr + pid_b * stride_c_b + offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, H]
        encoder_hidden_states: [B, T, H]
        process_weight: [H, H]
        Returns: (processed_encoder: [B, T, H], processed_hidden: [B, I, H])
        """
        # Validate inputs
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D [B, dim, H]"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"
        assert hidden_states.dtype == encoder_hidden_states.dtype == process_weight.dtype, "All tensors must have same dtype"
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA for Triton"

        L = T + I

        # 1) Concatenate sequences into out_cat [B, L, H] using Triton
        out_cat = torch.empty((B, L, H), dtype=hidden_states.dtype, device=hidden_states.device)

        BLOCK_H = 64  # tile along hidden dimension for concat
        grid_concat = (B, L, triton.cdiv(H, BLOCK_H))
        triton.run(
            concat_sequences_kernel[grid_concat](
                encoder_hidden_states, hidden_states, out_cat,
                B, T, I, H,
                *encoder_hidden_states.stride(), *hidden_states.stride(),
                *out_cat.stride(),
                BLOCK_H=BLOCK_H,
                num_warps=4,
                num_stages=2,
            )
        )

        # 2) Prepare W_T = process_weight.T [H, H]
        W_T = process_weight.transpose(0, 1).contiguous()  # [H, H]

        # 3) Allocate output processed [B, L, H] and perform GEMM with Triton
        processed = torch.empty((B, L, H), dtype=out_cat.dtype, device=out_cat.device)

        # 3D grid: (batch, tiles over M=L, tiles over N=H)
        BLOCK_M = 64   # sequence tiles
        BLOCK_N = 64   # output tiles (H)
        BLOCK_K = 64   # reduction tiles (H)
        grid = (B, triton.cdiv(L, BLOCK_M), triton.cdiv(H, BLOCK_N))

        triton.run(
            bmm_kernel[grid](
                out_cat, W_T, processed,
                B, L, H, H,  # M=L, N=H, K=H
                *out_cat.stride(),  # A strides: (B, M, K)
                *W_T.stride(),      # B strides: (K, N)
                *processed.stride(),# C strides: (B, M, N)
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4,
                num_stages=2,
            )
        )

        # 4) Split outputs back into encoder and hidden streams
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
