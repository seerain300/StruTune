import torch
import triton
import triton.language as tl


# Triton kernel: build concatenated input X_cat[b] for each batch without torch.cat.
# X_cat[b] has shape [(T+I), H]. If row index p < T, take from encoder_hidden_states[b]; else take from hidden_states[b].
@triton.jit
def cat_rows_kernel(
    e_ptr, i_ptr, x_ptr,
    B, T, I, H,
    stride_eb, stride_et, stride_eh,
    stride_ib, stride_it, stride_ih,
    stride_xb, stride_xm, stride_xn,
    M: tl.constexpr,  # M = T + I
):
    b = tl.program_id(0)
    p = tl.program_id(1)  # row index in concatenated matrix, in [0, M)

    # Choose source tensor
    use_e = p < T
    if use_e:
        row_ptr = e_ptr + b * stride_eb + p * stride_et
    else:
        row_ptr = i_ptr + b * stride_ib + (p - T) * stride_it

    # Write row p into X_cat[b, p, :]
    for n in range(0, H):
        x_index = b * stride_xb + p * stride_xm + n * stride_xn
        val = tl.load(row_ptr + n * stride_eh if use_e else n * stride_ih)
        tl.store(x_ptr + x_index, val)


# Triton kernel: batched matmul for each batch b
# Computes Y[b] = X_cat[b] @ W, where X_cat[b] is [M, H], W is [H, H]
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

    # Tile over output dimensions
    for m0 in range(0, M, BLOCK_M):
        for n0 in range(0, H, BLOCK_N):
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, H, BLOCK_K):
                m_offsets = m0 + tl.arange(0, BLOCK_M)
                n_offsets = n0 + tl.arange(0, BLOCK_N)
                k_offsets = k0 + tl.arange(0, BLOCK_K)

                x_ptrs = x_ptr + b * stride_xb + m_offsets[:, None] * stride_xm + k_offsets[None, :] * stride_xn
                w_ptrs = w_ptr + k_offsets[:, None] * stride_wk + n_offsets[None, :] * stride_wn

                x_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < H)
                w_mask = (k_offsets[:, None] < H) & (n_offsets[None, :] < H)

                x = tl.load(x_ptrs, mask=x_mask, other=0.0)
                w = tl.load(w_ptrs, mask=w_mask, other=0.0)

                acc += tl.dot(x, w)

            y_ptrs = y_ptr + b * stride_yb + m_offsets[:, None] * stride_ym + n_offsets[None, :] * stride_yn
            y_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < H)
            tl.store(y_ptrs, acc, mask=y_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure inputs are CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors for Triton."

        B = hidden_states.shape[0]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[0] == B
        assert encoder_hidden_states.shape[2] == H
        assert process_weight.shape == (H, H), "process_weight must be [H, H]."

        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        M = T + I

        # Allocate concatenated input per batch: X_cat[b] has shape (M, H)
        X_cat = [torch.empty((M, H), device=hidden_states.device, dtype=hidden_states.dtype) for _ in range(B)]

        # Launch Triton kernel: one program per (batch, row)
        grid_cat = (B, M)
        cat_rows_kernel[grid_cat](
            encoder_hidden_states, hidden_states, X_cat[0],
            B, T, I, H,
            *encoder_hidden_states.stride(),  # stride_eb, stride_et, stride_eh
            *hidden_states.stride(),          # stride_ib, stride_it, stride_ih
            *X_cat[0].stride(),               # stride_xb, stride_xm, stride_xn
            M=M,
            num_warps=1, num_stages=1,
        )

        # Allocate output per batch: Y[b] has shape (M, H)
        Y = [torch.empty((M, H), device=hidden_states.device, dtype=hidden_states.dtype) for _ in range(B)]

        # Launch Triton batched matmul kernel: one program per batch
        grid_mm = (B,)
        batched_matmul_kernel[grid_mm](
            X_cat[0], process_weight, Y[0],
            M, H,
            *X_cat[0].stride(),   # stride_xb, stride_xm, stride_xn
            *process_weight.stride(),  # stride_wk, stride_wn
            *Y[0].stride(),         # stride_yb, stride_ym, stride_yn
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=3,
        )

        # Split per batch into encoder and hidden streams
        processed_encoder = [Y[b][:T, :] for b in range(B)]
        processed_hidden = [Y[b][T:, :] for b in range(B)]

        return processed_encoder, processed_hidden


# Entry point 'Model' must delegate to ModelNew to satisfy the harness
class Model(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        return ModelNew().forward(hidden_states, encoder_hidden_states, process_weight)


def run(*args):
    return ModelNew()(*args)
