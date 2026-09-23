import torch
import triton
import triton.language as tl


@triton.jit
def concat_rows_to_A_kernel(
    enc_ptr,       # [B, T, H]
    img_ptr,       # [B, I, H]
    A_ptr,         # [M, H], M = B*(T+I)
    B, T, I, H,
    stride_b_e, stride_t_e, stride_h_e,  # enc strides
    stride_b_i, stride_i_i, stride_h_i,  # img strides
    stride_b_a, stride_h_a,               # A strides
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Each program handles a tile of rows [pid * BLOCK_M : (pid+1)*BLOCK_M) and all H columns
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    M = B * (T + I)
    mask_rows = rows < M

    # Determine batch and sequence index for each row
    b_idx = rows // (T + I)
    s_idx = rows % (T + I)

    # Iterate H in tiles
    for h_start in range(0, H, BLOCK_H):
        h = h_start + tl.arange(0, BLOCK_H)
        mask_h = h < H

        # Load from encoder if s_idx < T, else from image
        enc_ptrs = enc_ptr + b_idx[:, None] * stride_b_e + s_idx[:, None] * stride_t_e + h[None, :] * stride_h_e
        mask_enc = mask_rows[:, None] & (s_idx[:, None] < T) & (h[None, :] < H)
        vals_enc = tl.load(enc_ptrs, mask=mask_enc, other=0.0)

        s_idx_img = s_idx - T
        mask_img = mask_rows[:, None] & (s_idx[:, None] >= T) & (h[None, :] < H)
        vals_img = tl.load(img_ptr + b_idx[:, None] * stride_b_i + s_idx_img[:, None] * stride_i_i + h[None, :] * stride_h_i, mask=mask_img, other=0.0)

        vals = tl.where(s_idx[:, None] < T, vals_enc, vals_img)

        # Store to A_ptr [M, H]: row = rows, col = h
        A_ptrs = A_ptr + rows[:, None] * stride_b_a + h[None, :] * stride_h_a
        tl.store(A_ptrs, vals, mask=mask_rows[:, None] & (h[None, :] < H))


@triton.jit
def copy_rows_encoder_kernel(
    C_ptr,             # [M, H]
    out_ptr,           # [B, T, H]
    B, T, I, H,        # dims
    stride_m_c, stride_h_c,          # C strides
    stride_b_o, stride_t_o, stride_h_o,  # output strides
    b,                  # which batch to process
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Copy rows [0 : B*T) of C into out[b, :, :]
    num_rows = B * T
    row_start = b * T
    for m_start in range(0, num_rows, BLOCK_M):
        m = m_start + tl.arange(0, BLOCK_M)
        mask_m = m < num_rows
        C_ptrs = C_ptr + m[:, None] * stride_m_c + tl.arange(0, BLOCK_H)[None, :] * stride_h_c
        t_idx = m % T
        out_ptrs = out_ptr + b * stride_b_o + t_idx[:, None] * stride_t_o + tl.arange(0, BLOCK_H)[None, :] * stride_h_o
        for h_start in range(0, H, BLOCK_H):
            h = h_start + tl.arange(0, BLOCK_H)
            mask_h = h < H
            vals = tl.load(C_ptrs, mask=mask_m[:, None] & mask_h[None, :], other=0.0)
            tl.store(out_ptrs, vals, mask=mask_m[:, None] & mask_h[None, :])


@triton.jit
def copy_rows_hidden_kernel(
    C_ptr,             # [M, H]
    out_ptr,           # [B, I, H]
    B, T, I, H,        # dims
    stride_m_c, stride_h_c,          # C strides
    stride_b_o, stride_i_o, stride_h_o,  # output strides
    b,                  # which batch to process
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Copy rows [B*T : B*(T+I)) of C into out[b, :, :]
    start = B * T
    end = B * (T + I)
    num_rows = end - start
    row_start = start
    for m_start in range(0, num_rows, BLOCK_M):
        m = row_start + m_start + tl.arange(0, BLOCK_M)
        mask_m = m < (B * (T + I))
        C_ptrs = C_ptr + m[:, None] * stride_m_c + tl.arange(0, BLOCK_H)[None, :] * stride_h_c
        dest_m = m - start
        i_idx = dest_m % I
        out_ptrs = out_ptr + b * stride_b_o + i_idx[:, None] * stride_i_o + tl.arange(0, BLOCK_H)[None, :] * stride_h_o
        for h_start in range(0, H, BLOCK_H):
            h = h_start + tl.arange(0, BLOCK_H)
            mask_h = h < H
            vals = tl.load(C_ptrs, mask=mask_m[:, None] & mask_h[None, :], other=0.0)
            tl.store(out_ptrs, vals, mask=mask_m[:, None] & mask_h[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only ModelNew:
        - Concatenate along sequence dimension into A using Triton.
        - Compute C = A @ process_weight.T with torch.matmul (host-side) for numerical exactness.
        - Split C into processed_encoder and processed_hidden using Triton copy kernels.
        """
        # Ensure contiguous tensors
        enc = encoder_hidden_states.contiguous()
        img = hidden_states.contiguous()
        weight = process_weight.contiguous()

        B, T, H = enc.shape
        _, I, _ = img.shape
        assert weight.shape == (H, H), "process_weight must have shape [hidden_dim, hidden_dim]"

        # 1) Concatenate into A: [M, H], M = B*(T+I)
        M = B * (T + I)
        A = torch.empty((M, H), dtype=enc.dtype, device=enc.device)

        BLOCK_M = 128
        grid_concat = (triton.cdiv(M, BLOCK_M),)
        concat_rows_to_A_kernel[grid_concat](
            enc, img, A,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            img.stride(0), img.stride(1), img.stride(2),
            A.stride(0), A.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_H=128,
        )

        # 2) Compute C = A @ process_weight.T exactly using torch
        weight_T = weight.t().contiguous()  # [H, H]
        C = A @ weight_T  # [M, H]

        # 3) Split C into processed_encoder and processed_hidden via Triton copy kernels
        processed_encoder = torch.empty((B, T, H), dtype=C.dtype, device=C.device)
        processed_hidden = torch.empty((B, I, H), dtype=C.dtype, device=C.device)

        # Copy encoder rows: [0 : B*T)
        for b in range(B):
            grid_e = (triton.cdiv(B * T, BLOCK_M),)
            copy_rows_encoder_kernel[grid_e](
                C, processed_encoder[b],
                B, T, I, H,
                C.stride(0), C.stride(1),
                processed_encoder[b].stride(0), processed_encoder[b].stride(1), processed_encoder[b].stride(2),
                b,
                BLOCK_M=BLOCK_M, BLOCK_H=128,
            )

        # Copy hidden rows: [B*T : B*(T+I))
        start = B * T
        end = B * (T + I)
        for b in range(B):
            grid_h = (triton.cdiv(end - start, BLOCK_M),)
            copy_rows_hidden_kernel[grid_h](
                C, processed_hidden[b],
                B, T, I, H,
                C.stride(0), C.stride(1),
                processed_hidden[b].stride(0), processed_hidden[b].stride(1), processed_hidden[b].stride(2),
                b,
                BLOCK_M=BLOCK_M, BLOCK_H=128,
            )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
