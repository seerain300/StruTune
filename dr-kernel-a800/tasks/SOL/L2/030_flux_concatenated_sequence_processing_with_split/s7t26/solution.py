import torch
import triton
import triton.language as tl


@triton.jit
def cat_rows_kernel(
    e_ptr, i_ptr, out_ptr,
    B, T, I, H,
    stride_eb, stride_et, stride_eh,
    stride_ib, stride_it, stride_ih,
    stride_ob, stride_om, stride_on,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B, M), where M = T + I
    b = tl.program_id(0)
    m = tl.program_id(1)  # row index in concatenated matrix

    # columns
    n_offsets = tl.arange(0, BLOCK_N)
    mask_cols = n_offsets < H

    # Determine source row: encoder if m < T, else image at row (m - T)
    from_encoder = m < T
    e_row_offset = b * stride_eb + m * stride_et
    i_row_offset = b * stride_ib + (m - T) * stride_it

    # Load selected row
    e_vals = tl.load(e_ptr + e_row_offset + n_offsets * stride_eh, mask=mask_cols, other=0.0)
    i_vals = tl.load(i_ptr + i_row_offset + n_offsets * stride_ih, mask=mask_cols, other=0.0)
    selected = tl.where(from_encoder, e_vals, i_vals)

    # Store into output
    out_row_offset = b * stride_ob + m * stride_om
    tl.store(out_ptr + out_row_offset + n_offsets * stride_on, selected, mask=mask_cols)


@triton.jit
def batched_matmul_kernel(
    x_ptr, w_ptr, y_ptr,
    M, H,
    stride_xb, stride_xm, stride_xn,
    stride_wk, stride_wn,
    stride_yb, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # grid: (B,) processes one batch per program
    b = tl.program_id(0)

    # Tiled GEMM: Y = X @ W
    for m0 in range(0, M, BLOCK_M):
        for n0 in range(0, H, BLOCK_N):
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, H, BLOCK_K):
                m_offsets = m0 + tl.arange(0, BLOCK_M)
                n_offsets = n0 + tl.arange(0, BLOCK_N)
                k_offsets = k0 + tl.arange(0, BLOCK_K)

                mask_m = m_offsets < M
                mask_n = n_offsets < H
                mask_k = k_offsets < H

                # A tile: x[b, m, k]
                a_ptrs = x_ptr + b * stride_xb + m_offsets[:, None] * stride_xm + k_offsets[None, :] * stride_xn
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

                # B tile: w[k, n]
                b_ptrs = w_ptr + k_offsets[:, None] * stride_wk + n_offsets[None, :] * stride_wn
                b_w = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

                acc += tl.dot(a, b_w)

            # Store acc into y[b, m, n]
            y_ptrs = y_ptr + b * stride_yb + m_offsets[:, None] * stride_ym + n_offsets[None, :] * stride_yn
            tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]

        # Make inputs contiguous and float32
        e = encoder_hidden_states.contiguous().to(torch.float32)  # [B, T, H]
        i = hidden_states.contiguous().to(torch.float32)         # [B, I, H]
        w = process_weight.contiguous().to(torch.float32)        # [H, H]

        # Concatenated matrix: X_cat [B, M, H], M = T + I
        M = T + I
        X_cat = torch.empty((B, M, H), device=hidden_states.device, dtype=torch.float32)

        # Launch cat_rows_kernel: grid (B, M)
        BLOCK_N_cat = 128  # tile along hidden dim; masks ensure safety
        grid_cat = (B, M)
        cat_rows_kernel[grid_cat](
            e, i, X_cat,
            B, T, I, H,
            e.stride(0), e.stride(1), e.stride(2),
            i.stride(0), i.stride(1), i.stride(2),
            X_cat.stride(0), X_cat.stride(1), X_cat.stride(2),
            BLOCK_N=BLOCK_N_cat,
            num_warps=4, num_stages=2,
        )

        # Output buffer [B, M, H] for projection
        Y = torch.empty((B, M, H), device=hidden_states.device, dtype=torch.float32)

        # Launch batched_matmul_kernel: grid (B,)
        BLOCK_M_mm = 128
        BLOCK_N_mm = 128
        BLOCK_K_mm = 32
        grid_mm = (B,)
        batched_matmul_kernel[grid_mm](
            X_cat, w, Y,
            M, H,
            X_cat.stride(0), X_cat.stride(1), X_cat.stride(2),
            w.stride(0), w.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_M=BLOCK_M_mm, BLOCK_N=BLOCK_N_mm, BLOCK_K=BLOCK_K_mm,
            num_warps=4, num_stages=3,
        )

        # Split into encoder and hidden streams; cast back to original dtype
        processed_encoder = Y[:, :T, :].to(hidden_states.dtype)  # [B, T, H]
        processed_hidden = Y[:, T:, :].to(hidden_states.dtype)   # [B, I, H]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
