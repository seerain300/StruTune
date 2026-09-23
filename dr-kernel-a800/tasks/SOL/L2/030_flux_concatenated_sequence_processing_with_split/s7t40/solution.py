import torch
import triton
import triton.language as tl


@triton.jit
def cat_rows_kernel(
    ehs_ptr, hs_ptr, x_ptr,
    B, T, I, H,
    stride_eb, stride_et, stride_eh,
    stride_hb, stride_hi, stride_hh,
    stride_xb, stride_xm, stride_xh,
    BLOCK_M: tl.constexpr,
):
    # One program per (batch, row p)
    b = tl.program_id(0)
    p = tl.program_id(1)
    M = T + I

    # Offsets within a tile (this kernel handles 1 row per program)
    offs_m = p  # single row
    offs_h = tl.arange(0, BLOCK_M)  # vector of H-dim indices

    # Masks for boundary
    mask_m = (offs_m < M)  # always true since grid covers all M
    mask_h = offs_h < H

    # Decide source tensor
    use_encoder = p < T

    # Compute base pointers
    # ehs[b, p, h] or hs[b, p - T, h]
    if use_encoder:
        e_row_ptr = ehs_ptr + b * stride_eb + p * stride_et + offs_h * stride_eh
        # load from encoder
        x_row_ptr = x_ptr + b * stride_xb + offs_m * stride_xm + offs_h * stride_xh
        vals = tl.load(e_row_ptr, mask=mask_h, other=0.0)
    else:
        i_row = p - T
        h_row_ptr = hs_ptr + b * stride_hb + i_row * stride_hi + offs_h * stride_hh
        # load from hidden
        x_row_ptr = x_ptr + b * stride_xb + offs_m * stride_xm + offs_h * stride_xh
        vals = tl.load(h_row_ptr, mask=mask_h, other=0.0)

    # Store to X_cat[b, p, :]
    tl.store(x_row_ptr, vals, mask=mask_h)


@triton.jit
def batched_matmul_kernel(
    X_ptr, W_ptr, Y_ptr,
    M, N, K,
    stride_xb, stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_yb, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program handles a tile of the output for a single batch
    b = tl.program_id(0)

    # Output tile indices
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Masks for boundaries
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load A tile: X[b, offs_m, offs_k]
        a_ptrs = X_ptr + b * stride_xb + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        # Load B tile: W[offs_k, offs_n]
        w_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, w)

    # Store Y[b, offs_m, offs_n]
    y_ptrs = Y_ptr + b * stride_yb + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenate encoder_hidden_states and hidden_states along the sequence dimension using Triton.
        - Apply linear projection using Triton batched GEMM.
        - Split results back into separate encoder and image streams on host.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        M = T + I

        # 1) Concatenate with Triton
        # Allocate X_cat[B, M, H]
        X_cat = torch.empty((B, M, H), dtype=hidden_states.dtype, device=hidden_states.device)

        # Compute strides
        stride_eb, stride_et, stride_eh = encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2)
        stride_hb, stride_hi, stride_hh = hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2)
        stride_xb, stride_xm, stride_xh = X_cat.stride(0), X_cat.stride(1), X_cat.stride(2)

        # Launch cat kernel: grid = (B, M)
        grid_cat = (B, M)
        # We can choose a moderate BLOCK_M for H; since each program handles one row, we can use BLOCK_M=H.
        # Triton will broadcast scalars, and masks handle boundaries.
        cat_rows_kernel[grid_cat](
            encoder_hidden_states, hidden_states, X_cat,
            B, T, I, H,
            stride_eb, stride_et, stride_eh,
            stride_hb, stride_hi, stride_hh,
            stride_xb, stride_xm, stride_xh,
            BLOCK_M=H,
        )

        # 2) Linear projection with Triton: Y = X_cat @ process_weight
        # Ensure W is [H, H]
        W = process_weight  # already [H, H]
        # Allocate Y[B, M, H]
        Y = torch.empty((B, M, H), dtype=torch.float32, device=hidden_states.device)  # compute in fp32

        # Compute strides
        stride_xb, stride_xm, stride_xk = X_cat.stride(0), X_cat.stride(1), X_cat.stride(2)  # X_cat is [B, M, H], k = H
        stride_wk, stride_wn = W.stride(0), W.stride(1)  # W is [H, H]
        stride_yb, stride_ym, stride_yn = Y.stride(0), Y.stride(1), Y.stride(2)

        # Choose tiles: conservative for robustness
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid_mm = (B,)
        batched_matmul_kernel[grid_mm](
            X_cat, W, Y,
            M, H, H,
            stride_xb, stride_xm, stride_xk,
            stride_wk, stride_wn,
            stride_yb, stride_ym, stride_yn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # 3) Split on host: processed_encoder = Y[:, :T, :], processed_hidden = Y[:, T:, :]
        processed_encoder = Y[:, :T, :]
        processed_hidden = Y[:, T:, :]

        # Cast back to original dtype if needed
        if hidden_states.dtype != torch.float32:
            processed_encoder = processed_encoder.to(hidden_states.dtype)
            processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
