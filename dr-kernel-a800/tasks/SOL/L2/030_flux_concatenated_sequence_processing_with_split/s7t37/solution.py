import torch
import triton
import triton.language as tl

# Kernel 1: Build concatenated input X_cat[b] = [T+I, H] by selecting rows from
#           encoder_hidden_states[b, :, :] and hidden_states[b, :, :] without torch.cat.
# Each program writes one row p for a given batch b.
@triton.jit
def cat_rows_kernel(
    e_ptr,  # encoder_hidden_states [B, T, H]
    i_ptr,  # hidden_states [B, I, H]
    out_ptr,  # output [B, T+I, H], we will fill row p
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
    stride_eb, stride_et, stride_eh,
    stride_ib, stride_il, stride_ih,
    stride_ob, stride_om, stride_on,
):
    pid = tl.program_id(0)  # pid = b*(T+I) + p
    total = T + I
    b = pid // total
    p = pid % total

    # If p < T: take from encoder; else take from image
    is_encoder = p < T
    # Compute row pointer and load
    # Using mask to ensure we don't read past bounds
    if is_encoder:
        e_row_ptr = e_ptr + b * stride_eb + p * stride_et
        x_row = tl.load(e_row_ptr + tl.arange(0, H) * stride_eh, mask=tl.arange(0, H) < H, other=0.0)
    else:
        i_row = p - T  # index into image
        i_row_ptr = i_ptr + b * stride_ib + i_row * stride_il
        x_row = tl.load(i_row_ptr + tl.arange(0, H) * stride_ih, mask=tl.arange(0, H) < H, other=0.0)

    # Store to output [B, T+I, H], row p
    out_row_ptr = out_ptr + b * stride_ob + p * stride_om
    tl.store(out_row_ptr + tl.arange(0, H) * stride_on, x_row, mask=tl.arange(0, H) < H)


# Kernel 2: Batched GEMM Y[b] = X_cat[b] @ W, where W = process_weight [H, H].
# We implement Y[b, m, n] = sum_k X_cat[b, m, k] * W[k, n], for m in [0, M), n in [0, N).
# Grid is 1D over batches. Inside, we iterate over M and N in tiles.
@triton.jit
def batched_matmul_kernel(
    x_ptr,  # X_cat [B, M, H], but we load per-batch pointer as base + m*stride_xm
    w_ptr,  # process_weight [H, H]
    y_ptr,  # output [B, M, H]
    B, M, N, K,  # M = T+I, N=H, K=H
    stride_xb, stride_xm, stride_xk,
    stride_w0, stride_w1,
    stride_yb, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)

    # Output tiles
    # We'll loop over M and N in blocks and accumulate in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over N and M tiles
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        for m_start in range(0, M, BLOCK_M):
            m_offsets = m_start + tl.arange(0, BLOCK_M)
            # Accumulate over K dimension
            for k_start in range(0, K, BLOCK_K):
                k_offsets = k_start + tl.arange(0, BLOCK_K)

                # Load X[b, m, k] tile: shape (BLOCK_M, BLOCK_K)
                x_ptrs = x_ptr + b * stride_xb + m_offsets[:, None] * stride_xm + k_offsets[None, :] * stride_xk
                x_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
                x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

                # Load W[k, n] tile: shape (BLOCK_K, BLOCK_N)
                w_ptrs = w_ptr + k_offsets[:, None] * stride_w0 + n_offsets[None, :] * stride_w1
                w_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
                w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

                # Accumulate: acc[M_tile, N_tile] += sum over K_tile of x_tile * w_tile
                # x_tile: [BLOCK_M, BLOCK_K], w_tile: [BLOCK_K, BLOCK_N]
                acc += tl.dot(x_tile, w_tile)

    # Store result to y[b, m, n]
    # We store acc tile by tile
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        for m_start in range(0, M, BLOCK_M):
            m_offsets = m_start + tl.arange(0, BLOCK_M)
            y_ptrs = y_ptr + b * stride_yb + m_offsets[:, None] * stride_ym + n_offsets[None, :] * stride_yn
            y_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
            tl.store(y_ptrs, acc, mask=y_mask)  # acc is float32


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation of:
          concatenated = cat([encoder_hidden_states, hidden_states], dim=1)
          processed = concatenated @ process_weight.T
          processed_encoder = processed[:, :text_seq_len, :]
          processed_hidden = processed[:, text_seq_len:, :]

        Returns:
          (processed_encoder, processed_hidden)
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton."

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        K = H  # since process_weight is [H, H]

        # 1) Build X_cat[b] = [T+I, H] per batch using Triton kernel (no torch.cat)
        X_cat = [torch.empty((T + I, H), device=hidden_states.device, dtype=torch.float32) for _ in range(B)]

        grid_cat = (B * (T + I),)
        cat_rows_kernel[grid_cat](
            encoder_hidden_states, hidden_states, X_cat[0],  # pass pointers; X_cat is a list of tensors
            B, T, I, H,
            *encoder_hidden_states.stride(),
            *hidden_states.stride(),
            *X_cat[0].stride(),
            num_warps=2, num_stages=2,
        )
        # Note: In Triton, we can pass each batch's tensor separately by launching once per batch:
        # To support multiple batches, we can iterate b and set grid to (B*(T+I),). However Triton requires static grid,
        # so we launch per batch loop explicitly using Python:
        for b in range(B):
            X_cat[b] = torch.empty((T + I, H), device=hidden_states.device, dtype=torch.float32)
            cat_rows_kernel[(T + I,)](
                encoder_hidden_states[b], hidden_states[b], X_cat[b],
                1, T, I, H,  # dummy B here; we pass full strides anyway
                *encoder_hidden_states.stride(), *hidden_states.stride(),
                *X_cat[b].stride(),
                b, T, I, H,
                num_warps=2, num_stages=2,
            )

        # 2) Compute Y[b] = X_cat[b] @ process_weight using Triton GEMM
        Y = [torch.empty((T + I, H), device=hidden_states.device, dtype=torch.float32) for _ in range(B)]

        for b in range(B):
            # Strides
            stride_xb, stride_xm, stride_xk = X_cat[b].stride()
            stride_w0, stride_w1 = process_weight.stride()
            stride_yb, stride_ym, stride_yn = Y[b].stride()

            # Choose tile sizes; H can be up to 4096, so moderate tiles are fine
            BLOCK_M = 64
            BLOCK_N = 64
            BLOCK_K = 32

            grid_mm = (1,)  # one program per batch
            batched_matmul_kernel[grid_mm](
                X_cat[b], process_weight, Y[b],
                1, T + I, H, H,
                stride_xb, stride_xm, stride_xk,
                stride_w0, stride_w1,
                stride_yb, stride_ym, stride_yn,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )

        # 3) Split results (host slicing, allowed)
        processed_encoder = [Y[b][:T, :] for b in range(B)]
        processed_hidden = [Y[b][T:, :] for b in range(B)]

        # Cast back to original dtype
        processed_encoder = [pe.to(hidden_states.dtype) for pe in processed_encoder]
        processed_hidden = [ph.to(hidden_states.dtype) for ph in processed_hidden]

        return processed_encoder[0], processed_hidden[0]


def run(*args):
    return ModelNew()(*args)
