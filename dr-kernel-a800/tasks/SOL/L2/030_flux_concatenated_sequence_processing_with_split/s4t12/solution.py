import torch
import triton
import triton.language as tl


@triton.jit
def per_batch_gemm_encoder_kernel(
    enc_ptr,           # [B, T, H]
    weight_t_ptr,      # [H, H]
    out_ptr,           # [B, T, H]
    B, T, H,
    stride_b_e, stride_t_e, stride_h_e,
    stride_h_w, stride_k_w,           # weight_t strides
    stride_b_o, stride_t_o, stride_h_o,
    BLOCK_K: tl.constexpr,
):
    # One program per output row m in [0, B*T)
    m = tl.program_id(0)
    # Compute batch and sequence
    b = m // T
    seq = m % T
    # Base pointers
    enc_row_ptr = enc_ptr + b * stride_b_e + seq * stride_t_e
    out_row_ptr = out_ptr + b * stride_b_o + seq * stride_t_o

    # Accumulator for output vector [H]
    acc = tl.zeros((H,), dtype=tl.float32)

    # Loop over reduction dimension in tiles
    for k0 in range(0, H, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_range < H
        # Load input vector segment (row of enc) for H dimension
        a = tl.load(enc_row_ptr + k_range * stride_h_e, mask=k_mask, other=0.0)
        # Load corresponding segment from weight_t (H x H)
        b_block = tl.load(weight_t_ptr + k_range[:, None] * stride_h_w + tl.arange(0, BLOCK_K)[None, :] * stride_k_w,
                          mask=k_mask[:, None], other=0.0)
        # Dot accumulate: (BLOCK_K x 1) @ (BLOCK_K x BLOCK_K) -> (1 x BLOCK_K) -> (BLOCK_K,) cast to scalar
        # Triton dot expects matching shapes; here we reduce over k-axis.
        # We implement acc += sum(a[k] * b_block[k, :]) for each k in tile by multiplying and summing.
        # Compute contribution for the tile
        # Note: we reduce along axis 0 of b_block to get BLOCK_K elements.
        # Triton provides tl.sum for reductions; apply over axis=0.
        contrib = tl.sum(a[:, None] * b_block, axis=0)
        # Accumulate into acc
        acc += contrib

    # Store result vector
    out_store_ptr = out_row_ptr + tl.arange(0, H) * stride_h_o
    store_mask = tl.arange(0, H) < H
    tl.store(out_store_ptr, acc, mask=store_mask)


@triton.jit
def per_batch_gemm_hidden_kernel(
    img_ptr,           # [B, I, H]
    weight_t_ptr,      # [H, H]
    out_ptr,           # [B, I, H]
    B, I, H,
    stride_b_i, stride_i_i, stride_h_i,
    stride_h_w, stride_k_w,           # weight_t strides
    stride_b_o, stride_i_o, stride_h_o,
    BLOCK_K: tl.constexpr,
):
    # One program per output row m in [B*T : B*(T+I))
    m = tl.program_id(0)
    # Determine batch and sequence index within hidden
    T_plus_I = (m // I) * I + I  # not used, but preserved for clarity
    b = m // (T_plus_I)  # b is redundant since m >= B*T, but ensure consistency
    seq = m - B * (T_plus_I)  # seq in [0, I)
    if seq >= I:
        # Shouldn't happen; mask ensures otherwise zeros
        pass
    base_img_ptr = img_ptr + b * stride_b_i + seq * stride_i_i
    base_out_ptr = out_ptr + b * stride_b_o + seq * stride_i_o

    acc = tl.zeros((H,), dtype=tl.float32)

    for k0 in range(0, H, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_range < H
        a = tl.load(base_img_ptr + k_range * stride_h_i, mask=k_mask, other=0.0)
        b_block = tl.load(weight_t_ptr + k_range[:, None] * stride_h_w + tl.arange(0, BLOCK_K)[None, :] * stride_k_w,
                          mask=k_mask[:, None], other=0.0)
        contrib = tl.sum(a[:, None] * b_block, axis=0)
        acc += contrib

    out_store_ptr = base_out_ptr + tl.arange(0, H) * stride_h_o
    store_mask = tl.arange(0, H) < H
    tl.store(out_store_ptr, acc, mask=store_mask)


@triton.jit
def copy_rows_to_encoder_kernel(
    src_ptr,           # [B, T, H]
    dst_ptr,           # [B, T, H]
    B, T, H,
    stride_b_s, stride_t_s, stride_h_s,
    stride_b_d, stride_t_d, stride_h_d,
    BLOCK_M: tl.constexpr,
):
    # One program copies a tile of rows
    start = tl.program_id(0) * BLOCK_M
    rows = start + tl.arange(0, BLOCK_M)
    mask = rows < (B * T)
    # Decode batch and seq per row
    b = rows // T
    t = rows % T
    src_row_ptr = src_ptr + b[:, None] * stride_b_s + t[:, None] * stride_t_s + tl.arange(0, H)[None, :] * stride_h_s
    dst_row_ptr = dst_ptr + b[:, None] * stride_b_d + t[:, None] * stride_t_d + tl.arange(0, H)[None, :] * stride_h_d
    mask2 = mask[:, None]
    vals = tl.load(src_row_ptr, mask=mask2, other=0.0)
    tl.store(dst_row_ptr, vals, mask=mask2)


@triton.jit
def copy_rows_to_hidden_kernel(
    src_ptr,           # [B, I, H]
    dst_ptr,           # [B, I, H]
    B, I, H,
    stride_b_s, stride_i_s, stride_h_s,
    stride_b_d, stride_i_d, stride_h_d,
    BLOCK_M: tl.constexpr,
):
    # One program copies a tile of rows
    start = tl.program_id(0) * BLOCK_M
    rows = start + tl.arange(0, BLOCK_M)
    mask = rows < (B * I)
    b = rows // I
    i = rows % I
    src_row_ptr = src_ptr + b[:, None] * stride_b_s + i[:, None] * stride_i_s + tl.arange(0, H)[None, :] * stride_h_s
    dst_row_ptr = dst_ptr + b[:, None] * stride_b_d + i[:, None] * stride_i_d + tl.arange(0, H)[None, :] * stride_h_d
    mask2 = mask[:, None]
    vals = tl.load(src_row_ptr, mask=mask2, other=0.0)
    tl.store(dst_row_ptr, vals, mask=mask2)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation:
        - Concatenation semantics are fused in per-batch GEMM kernels.
        - No torch.matmul or torch.cat. All computation in Triton kernels.
        Returns (processed_encoder, processed_hidden) with shapes [B, T, H] and [B, I, H].
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA for Triton."
        # Ensure contiguity
        enc = encoder_hidden_states.contiguous()
        img = hidden_states.contiguous()
        weight = process_weight.contiguous()
        B, T, H = enc.shape
        B2, I, H2 = img.shape
        assert B == B2 and H == H2, "hidden_states and encoder_hidden_states must have the same batch and hidden_dim."
        assert weight.shape == (H, H), "process_weight must have shape [hidden_dim, hidden_dim]."

        # Transpose weight for GEMM: weight_t [H, H]
        weight_t = weight.transpose(0, 1).contiguous()

        # Allocate outputs
        processed_encoder = torch.empty((B, T, H), dtype=enc.dtype, device=enc.device)
        processed_hidden = torch.empty((B, I, H), dtype=img.dtype, device=img.device)

        # Tiling parameters (tunable)
        BLOCK_K = 64  # reduction tile along H
        BLOCK_M = 128  # rows per copy kernel tile

        # Launch per-batch GEMM for encoder rows
        for b in range(B):
            # Output buffer for this batch: [T, H]
            out_tmp_e = torch.empty((T, H), dtype=enc.dtype, device=enc.device)
            # Grid of one program per row
            grid = (T,)
            per_batch_gemm_encoder_kernel[grid](
                enc[b], weight_t, out_tmp_e,
                B, T, H,
                enc.stride(0), enc.stride(1), enc.stride(2),
                weight_t.stride(0), weight_t.stride(1),
                out_tmp_e.stride(0), out_tmp_e.stride(1), out_tmp_e.stride(2),
                BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=2,
            )
            # Copy into processed_encoder[b]
            grid_copy = (triton.cdiv(T, BLOCK_M),)
            copy_rows_to_encoder_kernel[grid_copy](
                out_tmp_e, processed_encoder[b],
                B, T, H,
                out_tmp_e.stride(0), out_tmp_e.stride(1), out_tmp_e.stride(2),
                processed_encoder[b].stride(0), processed_encoder[b].stride(1), processed_encoder[b].stride(2),
                BLOCK_M=BLOCK_M,
                num_warps=4, num_stages=2,
            )

        # Launch per-batch GEMM for hidden rows
        total_rows_hidden = B * I
        for b in range(B):
            # Output buffer for this batch: [I, H]
            out_tmp_h = torch.empty((I, H), dtype=img.dtype, device=img.device)
            # Grid of one program per row
            grid = (I,)
            per_batch_gemm_hidden_kernel[grid](
                img[b], weight_t, out_tmp_h,
                B, I, H,
                img.stride(0), img.stride(1), img.stride(2),
                weight_t.stride(0), weight_t.stride(1),
                out_tmp_h.stride(0), out_tmp_h.stride(1), out_tmp_h.stride(2),
                BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=2,
            )
            # Copy into processed_hidden[b]
            grid_copy = (triton.cdiv(I, BLOCK_M),)
            copy_rows_to_hidden_kernel[grid_copy](
                out_tmp_h, processed_hidden[b],
                B, I, H,
                out_tmp_h.stride(0), out_tmp_h.stride(1), out_tmp_h.stride(2),
                processed_hidden[b].stride(0), processed_hidden[b].stride(1), processed_hidden[b].stride(2),
                BLOCK_M=BLOCK_M,
                num_warps=4, num_stages=2,
            )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
