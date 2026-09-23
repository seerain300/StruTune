import torch
import triton
import triton.language as tl


# Triton kernel: build concatenated input X_cat[b] with shape [(T+I), H]
# For each batch b and sequence position p in [0, T+I):
#   if p < T: row from encoder_hidden_states[b]
#   else:     row from hidden_states[b]
@triton.jit
def cat_rows_kernel(
    e_ptr, i_ptr, y_ptr,
    B, T, I, H,
    stride_eb, stride_et, stride_eh,
    stride_ib, stride_it, stride_ih,
    stride_yb, stride_ym, stride_yn,
    M: tl.constexpr,  # M = T + I
):
    b = tl.program_id(0)
    p = tl.program_id(1)

    if (b >= B) or (p >= M):
        return

    # Determine source tensor
    is_encoder = p < T

    # Compute base pointers for the source row
    if is_encoder:
        row_ptr = e_ptr + b * stride_eb + p * stride_et
    else:
        row_ptr = i_ptr + b * stride_ib + (p - T) * stride_it

    # Destination pointer in y for row p
    y_row_ptr = y_ptr + b * stride_yb + p * stride_ym

    # Copy the entire hidden dimension
    cols = tl.arange(0, H)
    vals = tl.load(row_ptr + cols * (stride_eh if is_encoder else stride_ih))
    tl.store(y_row_ptr + cols * stride_yn, vals)


# Triton kernel: batched GEMM Y[b] = X_cat[b] @ W, where W has shape [H, H]
# X_cat[b] is [M, H], W is [H, H], Y[b] is [M, H]
@triton.jit
def batched_matmul_kernel(
    x_ptr, w_ptr, y_ptr,
    M, H,
    stride_xb, stride_xm, stride_xn,
    stride_wk, stride_wn,
    stride_yb, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)

    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Tile over K (hidden dimension)
    for k0 in range(0, H, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)

        # Loop over M tiles
        for m0 in range(0, M, BLOCK_M):
            m_idx = m0 + tl.arange(0, BLOCK_M)

            # Load A tile: X[b, m, k]
            a_ptrs = x_ptr + b * stride_xb + m_idx[:, None] * stride_xm + k_idx[None, :] * stride_xn
            a_mask = (m_idx[:, None] < M) & (k_idx[None, :] < H)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

            # Load W tile: W[k, n]
            n_idx = tl.arange(0, BLOCK_N)
            w_ptrs = w_ptr + k_idx[:, None] * stride_wk + n_idx[None, :] * stride_wn
            w_mask = (k_idx[:, None] < H) & (n_idx[None, :] < H)
            w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

            # Accumulate
            acc += tl.dot(a, w)

    # Store result to Y[b, :, :]
    m_out = tl.arange(0, BLOCK_M)
    n_out = tl.arange(0, BLOCK_N)
    y_ptrs = y_ptr + b * stride_yb + m_out[:, None] * stride_ym + n_out[None, :] * stride_yn
    y_mask = (m_out[:, None] < M) & (n_out[None, :] < H)
    tl.store(y_ptrs, acc, mask=y_mask)


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

        # Allocate concatenated input per batch: X_cat[b] of shape [M, H]
        X_cat = [torch.empty((M, H), device=hidden_states.device, dtype=hidden_states.dtype) for _ in range(B)]

        # Launch cat kernel: grid = (B, M)
        grid_cat = (B, M)

        e = encoder_hidden_states
        i = hidden_states

        # Strides for encoder and hidden
        stride_eb, stride_et, stride_eh = e.stride(0), e.stride(1), e.stride(2)
        stride_ib, stride_it, stride_ih = i.stride(0), i.stride(1), i.stride(2)
        # Strides for X_cat
        stride_yb, stride_ym, stride_yn = X_cat[0].stride(0), X_cat[0].stride(1), X_cat[0].stride(2)

        for b in range(B):
            cat_rows_kernel[grid_cat](
                e[b], i[b], X_cat[b],
                B, T, I, H,
                stride_eb, stride_et, stride_eh,
                stride_ib, stride_it, stride_ih,
                stride_yb, stride_ym, stride_yn,
                M=M,
                num_warps=1, num_stages=1,
            )

        # Allocate output per batch: Y[b] of shape [M, H] in float32 for compute
        Y = [torch.empty((M, H), device=hidden_states.device, dtype=torch.float32) for _ in range(B)]

        # Launch batched matmul kernel: one program per batch
        grid_mm = (B,)
        # Tile sizes: tuneable defaults
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32

        w = process_weight  # [H, H]
        # Ensure weight is float32 for compute
        w = w.to(torch.float32)

        for b in range(B):
            x_b = X_cat[b]
            y_b = Y[b]

            # Strides
            stride_xb, stride_xm, stride_xn = x_b.stride(0), x_b.stride(1), x_b.stride(2)
            stride_wk, stride_wn = w.stride(0), w.stride(1)
            stride_yb, stride_ym, stride_yn = y_b.stride(0), y_b.stride(1), y_b.stride(2)

            batched_matmul_kernel[grid_mm](
                x_b, w, y_b,
                M, H,
                stride_xb, stride_xm, stride_xn,
                stride_wk, stride_wn,
                stride_yb, stride_ym, stride_yn,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )

        # Split results into encoder and hidden streams (host slicing only, no torch ops)
        processed_encoder = [y_b[:T, :] for y_b in Y]
        processed_hidden = [y_b[T:, :] for y_b in Y]

        # Cast back to original dtype to match original function
        processed_encoder = [pe.to(hidden_states.dtype) for pe in processed_encoder]
        processed_hidden = [ph.to(hidden_states.dtype) for ph in processed_hidden]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
