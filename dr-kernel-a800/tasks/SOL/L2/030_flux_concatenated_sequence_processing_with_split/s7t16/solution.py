import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    X_ptr, W_ptr, Y_ptr,
    M, H,
    stride_xb, stride_xm, stride_xn,
    stride_wk, stride_wn,
    stride_yb, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program ids: grid=(B, ceil_div(M, BLOCK_M), ceil_div(H, BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # tile offsets
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    # bounds masks
    m_mask = m_offsets < M
    n_mask = n_offsets < H

    # accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K = H in steps of BLOCK_K
    for k_start in range(0, H, BLOCK_K):
        k = k_start + k_offsets
        k_mask = k < H

        # load X[b, m, k] -> [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xm + k[None, :] * stride_xn
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # load W[k, n] -> [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + k[:, None] * stride_wk + n_offsets[None, :] * stride_wn
        w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        # accumulate
        acc += tl.dot(x, w)

    # store Y[b, m, n] = acc
    y_ptrs = Y_ptr + b * stride_yb + m_offsets[:, None] * stride_ym + n_offsets[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def _grid_matmul(B, M, H, BLOCK_M, BLOCK_N):
    return (B, triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be CUDA for Triton."

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]

        # Concatenate along sequence dimension using torch (allowed; heavy op is Triton matmul)
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # shape [B, T+I, H]

        # Allocate output Y per batch in fp32 for numerical stability
        M = T + I
        Y = [None] * B

        for b in range(B):
            x_b = concatenated[b]  # [M, H]
            w = process_weight     # [H, H]
            # Output Y[b] is [M, H], dtype float32
            y = torch.empty((M, H), dtype=torch.float32, device=hidden_states.device)

            # Launch Triton matmul kernel per batch
            grid = _grid_matmul(1, M, H, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
            batched_matmul_kernel[grid](
                x_b, w, y,
                M, H,
                x_b.stride(0), x_b.stride(0), x_b.stride(1),
                w.stride(0), w.stride(1),
                y.stride(0), y.stride(0), y.stride(1),
                BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                num_warps=4, num_stages=3
            )

            Y[b] = y  # [M, H] float32

        # Split results (host-side slicing)
        processed_encoder = [Y[b][:T, :] for b in range(B)]
        processed_hidden = [Y[b][T:, :] for b in range(B)]

        # Cast back to original dtype
        out_dtype = hidden_states.dtype
        processed_encoder = [pe.to(out_dtype) for pe in processed_encoder]
        processed_hidden = [ph.to(out_dtype) for ph in processed_hidden]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
