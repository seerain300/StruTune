import torch
import triton
import triton.language as tl


@triton.jit
def concat_sequences_kernel(
    E_ptr,            # *ptr to encoder_hidden_states: [B, T, H]
    H_ptr,            # *ptr to hidden_states: [B, I, H]
    Out_ptr,          # *ptr to out_cat: [B, L, H]
    B, T, I, H, L,    # ints: batch, text_len, img_len, hidden_dim, L = T + I
    stride_E_b, stride_E_t, stride_E_h,  # strides for E
    stride_H_b, stride_H_i, stride_H_h,  # strides for H
    stride_O_b, stride_O_l, stride_O_h,  # strides for Out
    BLOCK_H: tl.constexpr,
):
    # 3D grid: (batch, sequence position, hidden tiles)
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    # Compute pointers for Out[b, l, h]
    Out_ptrs = Out_ptr + pid_b * stride_O_b + pid_l * stride_O_l + offs_h * stride_O_h

    # Determine source: encoder for l < T, hidden for l >= T
    mask_l = pid_l < T
    if mask_l:
        E_ptrs = E_ptr + pid_b * stride_E_b + pid_l * stride_E_t + offs_h * stride_E_h
        vals = tl.load(E_ptrs, mask=mask_h, other=0.0)
    else:
        l_eff = pid_l - T
        H_ptrs = H_ptr + pid_b * stride_H_b + l_eff * stride_H_i + offs_h * stride_H_h
        vals = tl.load(H_ptrs, mask=mask_h, other=0.0)

    tl.store(Out_ptrs, vals, mask=mask_h)


@triton.jit
def bmm_kernel(
    A_ptr,    # *ptr to out_cat: [B, M=L, K=H]
    B_ptr,    # *ptr to W_T: [K=H, N=H]
    C_ptr,    # *ptr to processed: [B, M=L, N=H]
    B, M, N, K,  # dimensions (B not used, but passed for consistency)
    stride_A_b, stride_A_m, stride_A_k,   # strides for A: [B, M, K]
    stride_B_k, stride_B_n,               # strides for B: [K, N]
    stride_C_b, stride_C_m, stride_C_n,   # strides for C: [B, M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid: (batch, tiles over M, tiles over N)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # A[b, m, k] -> [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + pid_b * stride_A_b + (offs_m[:, None] * stride_A_m) + (offs_k[None, :] * stride_A_k)
        A_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(A_ptrs, mask=A_mask, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_K]

        # B[k, n] -> [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + (offs_k[:, None] * stride_B_k) + (offs_n[None, :] * stride_B_n)
        B_mask = mask_k[:, None] & mask_n[None, :]
        b = tl.load(B_ptrs, mask=B_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Fused multiply-add
        acc += tl.dot(a, b)

    # Store results to C[b, m, n]
    C_ptrs = C_ptr + pid_b * stride_C_b + (offs_m[:, None] * stride_C_m) + (offs_n[None, :] * stride_C_n)
    C_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, H]
        encoder_hidden_states: [B, T, H]
        process_weight: [H, H]
        Returns: (processed_encoder: [B, T, H], processed_hidden: [B, I, H])
        """
        # Basic validations
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D [B, dim, H]"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"
        assert hidden_states.dtype == encoder_hidden_states.dtype == process_weight.dtype, "All tensors must have the same dtype"
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA for Triton"

        L = T + I

        # 1) Concatenate sequences into out_cat [B, L, H] using Triton
        out_cat = torch.empty((B, L, H), dtype=hidden_states.dtype, device=hidden_states.device)

        BLOCK_H = 64
        grid_concat = (B, L, triton.cdiv(H, BLOCK_H))
        triton.run(
            concat_sequences_kernel[grid_concat](
                encoder_hidden_states, hidden_states, out_cat,
                B, T, I, H, L,
                *encoder_hidden_states.stride(), *hidden_states.stride(),
                *out_cat.stride(),
                BLOCK_H=BLOCK_H,
                num_warps=4,
                num_stages=2,
            )
        )

        # 2) Prepare W_T = process_weight.T [H, H]
        W_T = process_weight.t().contiguous()  # [H, H]

        # 3) Allocate output processed [B, L, H] and perform GEMM with Triton
        processed = torch.empty((B, L, H), dtype=out_cat.dtype, device=out_cat.device)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (B, triton.cdiv(L, BLOCK_M), triton.cdiv(H, BLOCK_N))
        triton.run(
            bmm_kernel[grid](
                out_cat, W_T, processed,
                B, L, H, H,  # M=L, N=H, K=H
                *out_cat.stride(),  # A strides: [B, M, K]
                *W_T.stride(),      # B strides: [K, N]
                *processed.stride(),# C strides: [B, M, N]
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4,
                num_stages=2,
            )
        )

        # 4) Split back into encoder and hidden streams
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
