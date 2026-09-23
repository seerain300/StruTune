import torch
import triton
import triton.language as tl


@triton.jit
def matmul_fused_concat_kernel(
    enc_ptr,       # *const T, [B, T, H]
    img_ptr,       # *const T, [B, I, H]
    weight_t_ptr,  # *const T, [H, H] (process_weight.T)
    C_ptr,         # *T, [M, H], M = B*(T+I)
    B, T, I, H,    # dims (int32)
    stride_b_e, stride_t_e, stride_h_e,   # strides for enc
    stride_b_i, stride_i_i, stride_h_i,   # strides for img
    stride_h_w, stride_k_w,                # strides for weight_t (H, K) with K=H
    stride_m_c, stride_n_c,                # strides for C
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over output rows (M) and columns (N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    mask_m = offs_m < B * (T + I)
    mask_n = offs_n < H

    # Compute batch and sequence indices for each output row
    b_idx = offs_m // (T + I)              # [BLOCK_M]
    seq_idx = offs_m % (T + I)             # [BLOCK_M]
    is_encoder = seq_idx < T               # [BLOCK_M]

    # Pointers for enc and img rows (length H)
    enc_row_ptrs = enc_ptr + b_idx[:, None] * stride_b_e + seq_idx[:, None] * stride_t_e + tl.arange(0, H)[None, :] * stride_h_e
    src_t = seq_idx                        # valid when is_encoder
    src_i = seq_idx - T                   # valid when not is_encoder
    img_row_ptrs = img_ptr + b_idx[:, None] * stride_b_i + src_i[:, None] * stride_i_i + tl.arange(0, H)[None, :] * stride_h_i

    # Load corresponding rows; masked by is_encoder
    a_vals = tl.load(enc_row_ptrs, mask=(mask_m[:, None] & is_encoder[:, None]), other=0.0)
    a_vals = tl.where(is_encoder[:, None], a_vals, tl.load(img_row_ptrs, mask=(mask_m[:, None] & ~is_encoder[:, None]), other=0.0))

    # Accumulator for output tile [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=a_vals.dtype)

    # Reduction over hidden_dim (K = H) in tiles of BLOCK_K
    for k0 in range(0, H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = offs_k < H

        # Load weight_t tile: [BLOCK_K, BLOCK_N], weight_t[k, n]
        w_ptrs = weight_t_ptr + offs_k[:, None] * stride_h_w + offs_n[None, :] * stride_k_w
        w_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Multiply and accumulate: (M,K) @ (K,N)
        acc += tl.dot(a_vals[:, None, :BLOCK_K], w_tile[None, :, :])

    # Store results to C[offs_m, offs_n]
    c_ptrs = C_ptr + offs_m[:, None] * stride_m_c + offs_n[None, :] * stride_n_c
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=mask_out)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Fuses concatenation, matmul, and splitting into Triton kernels.
        Returns (processed_encoder, processed_hidden) with shapes [B, T, H] and [B, I, H].
        """
        # Validate inputs
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be [B, L, H]"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        M = B * (T + I)

        # Ensure contiguity and device consistency
        enc = encoder_hidden_states.contiguous()
        img = hidden_states.contiguous()
        weight_t = process_weight.t().contiguous()  # [H, H], weight.T
        # We assume inputs are on the same device; if not, move weight to enc.device
        if weight_t.device != enc.device:
            weight_t = weight_t.to(enc.device)

        # Allocate output C [M, H]
        C = torch.empty((M, H), dtype=enc.dtype, device=enc.device)

        # Launch Triton kernel: grid over (M, H) tiles
        # Choose block sizes: typical values; for large H, consider 128; for moderate H, 64 works well.
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        matmul_fused_concat_kernel[grid](
            enc, img, weight_t, C,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            img.stride(0), img.stride(1), img.stride(2),
            weight_t.stride(0), weight_t.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Split C into processed_encoder [B, T, H] and processed_hidden [B, I, H]
        processed_encoder = C[:B * T].reshape(B, T, H)
        processed_hidden = C[B * T:].reshape(B, I, H)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
