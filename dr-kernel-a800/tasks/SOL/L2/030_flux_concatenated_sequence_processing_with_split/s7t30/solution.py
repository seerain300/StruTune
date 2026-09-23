import torch
import triton
import triton.language as tl


@triton.jit
def cat_rows_kernel(
    e_ptr,        # *ptr to encoder_hidden_states: [B, T, H]
    i_ptr,        # *ptr to hidden_states: [B, I, H]
    out_ptr,      # *ptr to output X_cat: [B, M, H], M = T + I
    B, T, I, H,   # sizes (int32)
    stride_eb, stride_et, stride_eh,  # strides for e
    stride_ib, stride_ih, stride_iw,  # strides for i
    stride_ob, stride_om, stride_oh,  # strides for out
):
    pid_b = tl.program_id(0)
    pid_p = tl.program_id(1)  # row index in [0, T + I)
    use_e = pid_p < T
    e_row_ptr = e_ptr + pid_b * stride_eb + pid_p * stride_et
    i_row_ptr = i_ptr + pid_b * stride_ib + (pid_p - T) * stride_ih
    out_row_ptr = out_ptr + pid_b * stride_ob + pid_p * stride_om

    # Load full row into a vector of length H (no masks; safe because we iterate exactly H elements)
    h_idx = tl.arange(0, H)
    val_e = tl.load(e_row_ptr + h_idx * stride_eh)
    val_i = tl.load(i_row_ptr + h_idx * stride_iw)

    # Select based on use_e (scalar); Triton will broadcast the scalar
    selected = tl.where(use_e, val_e, val_i)
    tl.store(out_row_ptr + h_idx * stride_oh, selected)


@triton.jit
def batched_matmul_kernel(
    x_ptr,  # pointer to X_cat[b] (we pass strides to emulate [M, K])
    w_ptr,  # pointer to process_weight [K, N]
    y_ptr,  # pointer to Y[b] (we pass strides to emulate [M, N])
    B, M, K, N,                    # sizes (M = T + I, K = N = H)
    stride_xb, stride_xm, stride_xk,  # strides for X_cat[b] considered as [M, K]
    stride_wk, stride_wn,              # strides for W[K, N]
    stride_yb, stride_ym, stride_yn,   # strides for Y[b] considered as [M, N]
):
    pid_b = tl.program_id(0)
    # Initialize accumulator Y[b] to zeros
    m_idx = tl.arange(0, M)
    n_idx = tl.arange(0, N)
    acc = tl.zeros((M, N), dtype=tl.float32)

    # Loop over K = H
    for k in range(0, K):
        # Load x_vec: [M] from X_cat[b, :, k]
        x_vec = tl.load(x_ptr + pid_b * stride_xb + m_idx * stride_xm + k * stride_xk)
        # Load w_vec: [N] from W[k, :]
        w_vec = tl.load(w_ptr + k * stride_wk + n_idx * stride_wn)
        # Outer product: acc += x_vec[:, None] * w_vec[None, :]
        acc += x_vec[:, None].to(tl.float32) * w_vec[None, :].to(tl.float32)

    # Store acc to Y[b]
    tl.store(y_ptr + pid_b * stride_yb + m_idx[:, None] * stride_ym + n_idx[None, :] * stride_yn, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure tensors are on CUDA
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        M = T + I

        # Allocate X_cat and Y in float32 for numerical stability
        X_cat = torch.empty((B, M, H), device=hidden_states.device, dtype=torch.float32)
        Y = torch.empty((B, M, H), device=hidden_states.device, dtype=torch.float32)

        # Launch cat_rows_kernel: grid over (batch, M rows)
        grid_cat = (B, M)
        cat_rows_kernel[grid_cat](
            encoder_hidden_states, hidden_states, X_cat,
            B, T, I, H,
            *encoder_hidden_states.stride(), *hidden_states.stride(),
            *X_cat.stride(),
            num_warps=2, num_stages=2,
        )

        # Launch batched_matmul_kernel: compute Y = X_cat @ process_weight
        grid_mm = (B,)
        batched_matmul_kernel[grid_mm](
            X_cat, process_weight, Y,
            B, M, H, H,
            X_cat.stride(0), X_cat.stride(1), X_cat.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            num_warps=2, num_stages=2,
        )

        # Split results into encoder and hidden streams (host slicing, no torch ops)
        processed_encoder = Y[:, :T, :]
        processed_hidden = Y[:, T:, :]

        # Cast back to original dtype of hidden_states if needed (match original behavior)
        processed_encoder = processed_encoder.to(hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
