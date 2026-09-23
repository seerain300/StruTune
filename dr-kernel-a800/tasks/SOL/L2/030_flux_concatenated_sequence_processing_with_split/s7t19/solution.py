import torch
import triton
import triton.language as tl


@triton.jit
def cat_rows_kernel(
    e_ptr, i_ptr, x_ptr,
    B, T, I, H,
    stride_eb, stride_et, stride_eh,
    stride_ib, stride_it, stride_ih,
    stride_xb, stride_xt, stride_xh,
):
    # program ids
    b = tl.program_id(0)  # batch
    p = tl.program_id(1)  # concatenated sequence position

    # total rows
    M = T + I

    # valid?
    # Triton uses masks for loads/stores, so we gate on b in range
    # p is in [0, M), always valid for grid.

    # decide source: first T rows from encoder, remaining from image
    src_from_e = p < T

    # compute source row indices
    e_row = p
    i_row = p - T

    # compute pointers
    # base pointers for this batch
    e_batch_ptr = e_ptr + b * stride_eb
    i_batch_ptr = i_ptr + b * stride_ib
    x_batch_ptr = x_ptr + b * stride_xb

    # load from encoder or image depending on src_from_e
    # if src_from_e:
    #   load e[b, p, :]
    # else:
    #   load i[b, p-T, :]
    if src_from_e:
        # e[b, p, :]
        e_row_ptr = e_batch_ptr + e_row * stride_et
        row_ptr = e_row_ptr
        # vector of column indices [0..H)
        cols = tl.arange(0, H)
        vals = tl.load(row_ptr + cols * stride_eh)  # assuming last dimension contiguous stride=1; use stride_eh if not
        # store into x[b, p, :]
        x_row_ptr = x_batch_ptr + p * stride_xt
        tl.store(x_row_ptr + cols * stride_xh, vals)
    else:
        # i[b, p-T, :]
        i_row_ptr = i_batch_ptr + i_row * stride_it
        row_ptr = i_row_ptr
        cols = tl.arange(0, H)
        vals = tl.load(row_ptr + cols * stride_ih)
        x_row_ptr = x_batch_ptr + p * stride_xt
        tl.store(x_row_ptr + cols * stride_xh, vals)


@triton.jit
def batched_matmul_kernel(
    x_ptr, w_ptr, y_ptr,
    M, H,
    stride_xb, stride_xm, stride_xn,
    stride_wk, stride_wn,
    stride_yb, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, ceil_div(M, BLOCK_M), ceil_div(H, BLOCK_N))
    b = tl.program_id(0)

    # Tile coordinates
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of X and Y
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # columns of X and Y (hidden dim)
    k_offsets = tl.arange(0, BLOCK_K)                   # reduction dim

    # Masks for boundaries
    m_mask = m_offsets < M
    n_mask = n_offsets < H

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension (H)
    for k in range(0, H, BLOCK_K):
        k_idx = k + k_offsets  # current reduction indices
        k_mask = k_idx < H

        # Pointers for X[b, m, k] and W[k, n]
        x_ptrs = x_ptr + b * stride_xb + m_offsets[:, None] * stride_xm + k_idx[None, :] * stride_xn
        w_ptrs = w_ptr + k_idx[:, None] * stride_wk + n_offsets[None, :] * stride_wn

        # Loads with masks
        x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w_vals = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        # Cast to float32 for accumulation
        x_vals = x_vals.to(tl.float32)
        w_vals = w_vals.to(tl.float32)

        # Accumulate
        acc += tl.dot(x_vals, w_vals)

    # Store results
    y_ptrs = y_ptr + b * stride_yb + m_offsets[:, None] * stride_ym + n_offsets[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation:
        - Concatenates encoder_hidden_states and hidden_states along sequence dim via Triton.
        - Applies linear projection via Triton batched GEMM.
        - Returns processed_encoder and processed_hidden slices.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton kernels."
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D tensors [B, L, H]."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        M = T + I

        # Allocate X_cat for each batch: [B, M, H]
        X = torch.empty((B, M, H), device=hidden_states.device, dtype=torch.float32)

        # Launch concatenation kernel
        grid_cat = (B, M)
        cat_rows_kernel[grid_cat](
            encoder_hidden_states, hidden_states, X,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            X.stride(0), X.stride(1), X.stride(2),
            num_warps=4, num_stages=2
        )

        # Allocate output Y for each batch: [B, M, H], float32
        Y = torch.empty((B, M, H), device=hidden_states.device, dtype=torch.float32)

        # Launch batched matmul kernel: X[b] @ process_weight
        grid_mm = (B, triton.cdiv(M, 64), triton.cdiv(H, 64))
        batched_matmul_kernel[grid_mm](
            X, process_weight, Y,
            M, H,
            X.stride(0), X.stride(1), X.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=3
        )

        # Split results: first T rows are encoder, remaining are image
        # Return slices as views (metadata ops)
        processed_encoder = [Y[b][:T, :] for b in range(B)]
        processed_hidden = [Y[b][T:, :] for b in range(B)]

        # Cast back to original dtype for consistency (hidden_states.dtype)
        out_dtype = hidden_states.dtype
        processed_encoder = [pe.to(out_dtype) for pe in processed_encoder]
        processed_hidden = [ph.to(out_dtype) for ph in processed_hidden]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
