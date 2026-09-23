import torch
import triton
import triton.language as tl


# Triton kernel: concatenate along sequence axis for a single batch.
# Inputs:
#   e_ptr: pointer to encoder_hidden_states[b] with shape [T, H]
#   i_ptr: pointer to hidden_states[b] with shape [I, H]
#   x_ptr: pointer to X_cat[b] with shape [(T+I), H]
#   B: batch size (unused here since we handle single b in forward; kept for completeness)
#   T: text seq len
#   I: img seq len
#   H: hidden dim
#   stride_eb, stride_et, stride_eh: strides for e
#   stride_ib, stride_it, stride_ih: strides for i
#   stride_xb, stride_xm, stride_xn: strides for x
@triton.jit
def cat_rows_kernel(e_ptr, i_ptr, x_ptr,
                     B, T, I, H,
                     stride_eb, stride_et, stride_eh,
                     stride_ib, stride_it, stride_ih,
                     stride_xb, stride_xm, stride_xn,
                     M):
    b = tl.program_id(0)
    p = tl.program_id(1)  # row index in concatenated sequence

    # Compute pointers for source and destination
    # Destination row offset in X_cat[b]
    x_row_ptr = x_ptr + b * stride_xb + p * stride_xm

    # If p < T, take from encoder_hidden_states[b]; else take from hidden_states[b]
    if p < T:
        src_ptr = e_ptr + b * stride_eb + p * stride_et
    else:
        src_ptr = i_ptr + b * stride_ib + (p - T) * stride_it

    # Load the entire row (length H) and store to X_cat
    # Create column indices
    cols = tl.arange(0, H)
    # Load with mask: columns are always within H, rows are guaranteed by M
    vals = tl.load(src_ptr + cols * stride_eh if p < T else src_ptr + cols * stride_ih)
    tl.store(x_row_ptr + cols * stride_xn, vals)


# Triton kernel: batched GEMM for each batch: Y[b] = X_cat[b] @ W, where W is [H, H]
@triton.jit
def batched_matmul_kernel(x_ptr, w_ptr, y_ptr,
                           M, N,
                           stride_xb, stride_xm, stride_xn,
                           stride_wk, stride_wn,  # W has shape [H, H] -> strides (k, n)
                           stride_yb, stride_ym, stride_yn,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    b = tl.program_id(0)

    # Tile indices
    for m in range(0, M, BLOCK_M):
        for n in range(0, N, BLOCK_N):
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in range(0, N, BLOCK_K):  # N == H
                m_offsets = m + tl.arange(0, BLOCK_M)
                n_offsets = n + tl.arange(0, BLOCK_N)
                k_offsets = k + tl.arange(0, BLOCK_K)

                m_mask = m_offsets < M
                n_mask = n_offsets < N
                k_mask = k_offsets < N  # since N == H and loops over N, this is fine

                # Load tiles: X[b, m_offsets, k_offsets], W[k_offsets, n_offsets]
                x_ptrs = x_ptr + b * stride_xb + m_offsets[:, None] * stride_xm + k_offsets[None, :] * stride_xn
                w_ptrs = w_ptr + k_offsets[:, None] * stride_wk + n_offsets[None, :] * stride_wn

                x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
                w_vals = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

                acc += tl.dot(x_vals, w_vals)

            # Store acc to Y[b, m_offsets, n_offsets]
            y_ptrs = y_ptr + b * stride_yb + m_offsets[:, None] * stride_ym + n_offsets[None, :] * stride_yn
            # Cast back to original dtype of y (assume float32 in forward)
            tl.store(y_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure CUDA tensors for Triton
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors for Triton kernels."
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "hidden_states and encoder_hidden_states must be 3D [B, L, H]."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert process_weight.shape == (H, H), "process_weight must be [H, H]."
        M = T + I

        # Allocate concatenated input per batch
        X_cat = [torch.empty((M, H), device=hidden_states.device, dtype=hidden_states.dtype) for _ in range(B)]

        # Launch cat kernel: grid = (B, M), one program per row
        grid_cat = (B, M)
        stride_eb, stride_et, stride_eh = encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2)
        stride_ib, stride_it, stride_ih = hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2)
        stride_xb, stride_xm, stride_xn = X_cat[0].stride(0), X_cat[0].stride(1), X_cat[0].stride(2)

        # Note: We pass B for signature but we only handle single b in forward; grid_cat(0) is batch id in Triton, grid_cat(1) is row index p
        for b in range(B):
            cat_rows_kernel[grid_cat](
                encoder_hidden_states[b], hidden_states[b], X_cat[b],
                B, T, I, H,
                stride_eb, stride_et, stride_eh,
                stride_ib, stride_it, stride_ih,
                stride_xb, stride_xm, stride_xn,
                M,
                num_warps=1, num_stages=1,
            )

        # Allocate output per batch
        Y = [torch.empty((M, H), device=hidden_states.device, dtype=torch.float32) for _ in range(B)]  # compute in fp32

        # Launch batched matmul kernel: grid = (B,)
        grid_mm = (B,)
        # Use H as N
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        for b in range(B):
            x_b = X_cat[b]  # dtype matches hidden_states
            w = process_weight
            y_b = Y[b]

            # Strides for current batch
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

        # Split results into encoder and hidden streams (host slicing only)
        processed_encoder = [Y[b][:T, :] for b in range(B)]
        processed_hidden = [Y[b][T:, :] for b in range(B)]

        # Cast back to original dtype to match original function
        processed_encoder = [pe.to(hidden_states.dtype) for pe in processed_encoder]
        processed_hidden = [ph.to(hidden_states.dtype) for ph in processed_hidden]

        return processed_encoder[0], processed_hidden[0]


def run(*args):
    return ModelNew()(*args)
