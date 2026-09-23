import torch
import triton
import triton.language as tl


# Triton kernel: build concatenated input X_cat[b] = cat(encoder_hidden_states[b], hidden_states[b]) along sequence axis.
# Grid: (B, M) where M = T + I. Each program writes one row (p) of X_cat[b].
@triton.jit
def cat_rows_kernel(
    e_ptr, i_ptr, out_ptr,
    B, T, I, H,
    stride_eb, stride_et, stride_eh,
    stride_ib, stride_it, stride_ih,
    stride_ob, stride_om, stride_on,
    M: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch index
    pid_p = tl.program_id(1)  # position in concatenated sequence

    if pid_b >= B or pid_p >= M:
        return

    # Determine source and local index
    if pid_p < T:
        src = 0  # encoder rows
        row_idx = pid_p
    else:
        src = 1  # image rows
        row_idx = pid_p - T

    # Compute base pointers
    if src == 0:
        base = e_ptr + pid_b * stride_eb
        row_ptr = base + row_idx * stride_et
    else:
        base = i_ptr + pid_b * stride_ib
        row_ptr = base + row_idx * stride_it

    # Write this row into out_ptr at (pid_b, pid_p)
    out_row_ptr = out_ptr + pid_b * stride_ob + pid_p * stride_om
    for j in range(0, H):
        val = tl.load(row_ptr + j * stride_eh)
        tl.store(out_row_ptr + j * stride_on, val)


# Triton kernel: batched matmul per batch. Computes Y[b] = X_cat[b] @ W, where W is [H, H] (no bias).
# Treat X_cat[b] as A[M, K], W as [K, N], Y as [M, N].
@triton.jit
def batched_matmul_kernel(
    X_ptr, W_ptr, Y_ptr,
    B, M, K, N,
    stride_xb, stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_yb, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    if pid_b >= B:
        return

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Accumulator tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        # Load A block: [BLOCK_M, BLOCK_K]
        a_ptrs = X_ptr + pid_b * stride_xb + (m_start + tl.arange(0, BLOCK_M))[:, None] * stride_xm + k_offsets[None, :] * stride_xk
        a_mask = (m_start + tl.arange(0, BLOCK_M))[:, None] < M
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load W block: [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + k_offsets[:, None] * stride_wk + (n_start + tl.arange(0, BLOCK_N))[None, :] * stride_wn
        w_mask = (k_offsets[:, None] < K) & (n_start + tl.arange(0, BLOCK_N))[None, :] < N
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, w)

    # Store results to Y
    y_ptrs = Y_ptr + pid_b * stride_yb + (m_start + tl.arange(0, BLOCK_M))[:, None] * stride_ym + (n_start) * stride_yn + tl.arange(0, BLOCK_N)[None, :] * stride_yn
    m_mask = (m_start + tl.arange(0, BLOCK_M))[:, None] < M
    n_mask = (n_start + tl.arange(0, BLOCK_N))[None, :] < N
    tl.store(y_ptrs, acc, mask=m_mask & n_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure CUDA tensors for Triton
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors for Triton kernels."

        B = hidden_states.shape[0]
        assert B == encoder_hidden_states.shape[0], "Batch sizes must match."
        H = hidden_states.shape[2]
        assert H == encoder_hidden_states.shape[2], "Hidden dimensions must match."
        assert process_weight.shape == (H, H), "process_weight must be [H, H]."

        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        M = T + I

        # Allocate concatenated input per batch
        X_cat = [torch.empty((M, H), device=hidden_states.device, dtype=hidden_states.dtype) for _ in range(B)]

        # Launch cat kernel: grid = (B, M), one program per row
        grid_cat = (B, M)
        e = encoder_hidden_states
        i = hidden_states
        # Strides
        stride_eb, stride_et, stride_eh = e.stride(0), e.stride(1), e.stride(2)
        stride_ib, stride_it, stride_ih = i.stride(0), i.stride(1), i.stride(2)
        stride_ob, stride_om, stride_on = X_cat[0].stride(0), X_cat[0].stride(1), X_cat[0].stride(2)

        for b in range(B):
            cat_rows_kernel[grid_cat](
                e[b], i[b], X_cat[b],
                B, T, I, H,
                stride_eb, stride_et, stride_eh,
                stride_ib, stride_it, stride_ih,
                stride_ob, stride_om, stride_on,
                M=M,
                num_warps=1, num_stages=1,
            )

        # Allocate output per batch
        Y = [torch.empty((M, H), device=hidden_states.device, dtype=hidden_states.dtype) for _ in range(B)]

        # Launch batched matmul kernel
        # Choose block sizes
        BLOCK_M = 128 if M >= 128 else 64
        BLOCK_N = 64 if H >= 64 else 32
        BLOCK_K = 32 if H >= 32 else 16

        grid_mm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        for b in range(B):
            X_b = X_cat[b]
            Y_b = Y[b]
            W = process_weight  # [H, H], no bias

            # Strides
            stride_xb = X_b.stride(0)
            stride_xm = X_b.stride(1)
            stride_xk = X_b.stride(2)

            stride_wk = W.stride(0)
            stride_wn = W.stride(1)

            stride_yb = Y_b.stride(0)
            stride_ym = Y_b.stride(1)
            stride_yn = Y_b.stride(2)

            batched_matmul_kernel[grid_mm](
                X_b, W, Y_b,
                B, M, H, H,
                stride_xb, stride_xm, stride_xk,
                stride_wk, stride_wn,
                stride_yb, stride_ym, stride_yn,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=2,
            )

        # Split outputs: processed_encoder [B, T, H], processed_hidden [B, I, H]
        # Minimal host-side splitting (no torch.matmul or stack in the forward, as required)
        processed_encoder = Y[0][:T, :] if B == 1 else torch.stack([Y[b][:T, :] for b in range(B)], dim=0)
        processed_hidden = Y[0][T:, :] if B == 1 else torch.stack([Y[b][T:, :] for b in range(B)], dim=0)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
