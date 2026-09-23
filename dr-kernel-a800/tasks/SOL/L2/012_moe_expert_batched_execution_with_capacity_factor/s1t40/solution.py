import torch
import triton
import triton.language as tl


@triton.jit
def bmm_triton_kernel(
    X_ptr,          # *f32 or *bf16, pointer to [B, H]
    W_ptr,          # *f32 or *bf16, pointer to [H, M]
    Y_ptr,          # *f32, pointer to [B, M]
    B: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    stride_xb, stride_xh,
    stride_wh, stride_wm,
    stride_yb, stride_ym,
    BLOCK_M: tl.constexpr,  # tile for M
    BLOCK_K: tl.constexpr,  # tile for reduction H
):
    # Compute Y[b, m] = sum_k X[b, k] * W[k, m] with b fixed (B=1 in our usage)
    b = 0
    m_start = tl.program_id(0) * BLOCK_M
    k_start = tl.program_id(1) * BLOCK_K

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    k_offsets = k_start + tl.arange(0, BLOCK_K)

    mask_m = m_offsets < M
    mask_k = k_offsets < H

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over reduction dimension K (H)
    for k0 in range(0, H, BLOCK_K):
        k_offsets_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets_k < H

        # Load X[b, k] -> 1D vector
        x_ptrs = X_ptr + b * stride_xb + k_offsets_k * stride_xh
        x = tl.load(x_ptrs, mask=mask_k, other=0.0)

        # Load W[k, m] -> 2D [BLOCK_K, BLOCK_M]
        w_ptrs = W_ptr + k_offsets_k[:, None] * stride_wh + m_offsets[None, :] * stride_wm
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)

        # Accumulate: acc += sum_k x[k] * w[k, m]
        acc += tl.sum(w * x[:, None], axis=0)

    # Store Y[b, m]
    y_ptrs = Y_ptr + b * stride_yb + m_offsets * stride_ym
    tl.store(y_ptrs, acc, mask=mask_m)


@triton.jit
def activation_silu_mul_kernel(
    Z_ptr,          # *f32, pointer to gate_out [M]
    U_ptr,          # *f32, pointer to up_out [M]
    Y_ptr,          # *f32, pointer to activated [M]
    M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Elementwise: Y = silu(Z) * U, where silu(Z) = Z * sigmoid(Z)
    for m in range(0, M, BLOCK):
        offsets = m + tl.arange(0, BLOCK)
        mask = offsets < M
        z = tl.load(Z_ptr + offsets, mask=mask, other=0.0)
        u = tl.load(U_ptr + offsets, mask=mask, other=0.0)
        s = 1.0 / (1.0 + tl.exp(-z))
        y = z * s * u
        tl.store(Y_ptr + offsets, y, mask=mask)


@triton.jit
def atomic_add_vec_kernel(
    out_ptr,         # *f32, pointer to result [num_tokens, hidden_size]
    add_ptr,         # *f32, pointer to vector to add [hidden_size]
    weights,         # scalar float32
    N,               # num_tokens
    H,               # hidden_size
    stride_out_n, stride_out_h,
    BLOCK: tl.constexpr,
):
    # Accumulate into out[row] += weights * add
    for n in range(0, N):
        for h in range(0, H, BLOCK):
            h_offsets = h + tl.arange(0, BLOCK)
            mask = h_offsets < H
            add_vals = tl.load(add_ptr + h_offsets, mask=mask, other=0.0)
            out_vals = tl.load(out_ptr + n * stride_out_n + h_offsets * stride_out_h, mask=mask, other=0.0)
            out_vals += weights * add_vals
            tl.store(out_ptr + n * stride_out_n + h_offsets * stride_out_h, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Triton-only forward: no torch.bmm, no torch.index_add, no torch.softmax on tensors.

        device = hidden_states.device
        num_tokens, hidden_size = hidden_states.shape


def run(*args):
    return ModelNew()(*args)
