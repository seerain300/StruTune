import torch
import triton
import triton.language as tl


@triton.jit
def concat_encoder_image_kernel(
    e_ptr,                # *const T: [B, T, H]
    i_ptr,                # *const T: [B, I, H]
    out_ptr,              # *T: [B, T+I, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    e_s0, e_s1, e_s2,     # strides for e_ptr
    i_s0, i_s1, i_s2,     # strides for i_ptr
    o_s0, o_s1, o_s2,     # strides for out_ptr
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # 3D launch: (batch, tiles over L=T+I, tiles over H)
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)        # shape [BLOCK_l], sequence positions in [0, T+I)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)        # shape [BLOCK_h], hidden dims

    L = T + I
    mask_l = l < L
    mask_h = h < H
    mask = mask_l[:, None] & mask_h[None, :]

    # For each (b, l, h), if l < T: load from encoder, else: load from image at (l - T)
    load_encoder = l < T

    # Compute addresses for encoder and image
    # e_ptr + b*e_s0 + l*e_s1 + h*e_s2
    # i_ptr + b*i_s0 + (l - T)*i_s1 + h*i_s2
    e_addrs = e_ptr + pid_b * e_s0 + l[:, None] * e_s1 + h[None, :] * e_s2
    i_addrs = i_ptr + pid_b * i_s0 + (l[:, None] - T) * i_s1 + h[None, :] * i_s2

    # Load with mask; for invalid positions, load zeros
    e_vals = tl.load(e_addrs, mask=mask & load_encoder[:, None], other=0.0)
    i_vals = tl.load(i_addrs, mask=mask & (~load_encoder)[:, None], other=0.0)

    vals = tl.where(load_encoder[:, None], e_vals, i_vals)  # shape [BLOCK_l, BLOCK_h]

    out_addrs = out_ptr + pid_b * o_s0 + l[:, None] * o_s1 + h[None, :] * o_s2
    tl.store(out_addrs, vals, mask=mask)


@triton.jit
def matmul_rowwise_kernel(
    a_ptr,        # *const T: [B, L, H], here L = T+I
    w_ptr,        # *const T: [H, H] weight (not transposed)
    c_ptr,        # *T: [B, L, H] output
    B: tl.int32, L: tl.int32, H: tl.int32,
    a_s0, a_s1, a_s2,      # strides for a_ptr
    w_s0, w_s1,            # strides for w_ptr (2D: [H, H])
    c_s0, c_s1, c_s2,      # strides for c_ptr
    BLOCK_k: tl.constexpr,
):
    # Grid: (B, L, tiles over H)
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    h_out = pid_h * BLOCK_k + tl.arange(0, BLOCK_k)  # output hidden indices
    mask_h = h_out < H

    acc = tl.zeros([BLOCK_k], dtype=tl.float32)

    # Loop over K (hidden dimension) in chunks
    for k in range(0, H, BLOCK_k):
        k_range = k + tl.arange(0, BLOCK_k)  # [BLOCK_k]
        mask_k = k_range < H

        # Load a_row: A[b, l, k_range]
        a_addrs = a_ptr + pid_b * a_s0 + pid_l * a_s1 + k_range * a_s2
        a_row = tl.load(a_addrs, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_k]

        # Load w_block: W[k_range, h_out] -> 2D [BLOCK_k, BLOCK_k]
        w_addrs = w_ptr + k_range[:, None] * w_s0 + h_out[None, :] * w_s1
        w_block = tl.load(w_addrs, mask=mask_k[:, None] & mask_h[None, :], other=0.0).to(tl.float32)  # [BLOCK_k, BLOCK_k]

        # Accumulate: dot(a_row, w_block)
        # a_row: [BLOCK_k], w_block: [BLOCK_k, BLOCK_k] -> [BLOCK_k]
        acc += tl.sum(a_row[:, None] * w_block, axis=0)

    # Store result to C[b, l, h_out]
    c_addrs = c_ptr + pid_b * c_s0 + pid_l * c_s1 + h_out * c_s2
    tl.store(c_addrs, acc, mask=mask_h)


@triton.jit
def copy_rows_kernel(
    src_ptr, dst_ptr,              # [B, L, H] and [B, out_L, H]
    B: tl.int32, L: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,
    dst_s0, dst_s1, dst_s2,
    ROW_START: tl.int32,           # starting row index in src to copy
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # 3D grid over (B, tiles over rows to copy, tiles over H)
    pid_b = tl.program_id(0)
    pid_rows = tl.program_id(1)
    pid_h = tl.program_id(2)

    l = ROW_START + pid_rows * BLOCK_l + tl.arange(0, BLOCK_l)  # source row indices to copy
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)

    mask_l = l < (ROW_START + L)
    mask_h = h < H
    mask = mask_l[:, None] & mask_h[None, :]

    src_addrs = src_ptr + pid_b * src_s0 + l[:, None] * src_s1 + h[None, :] * src_s2
    dst_addrs = dst_ptr + pid_b * dst_s0 + l[:, None] * dst_s1 + h[None, :] * dst_s2

    vals = tl.load(src_addrs, mask=mask, other=0.0)
    tl.store(dst_addrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension.
        - Applies linear projection using Triton GEMM.
        - Splits back into processed_encoder and processed_hidden.
        All tensor operations are performed by Triton kernels; no torch operations are used on tensors in host code.
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3 and process_weight.dim() == 2
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]

        device = hidden_states.device
        dtype = hidden_states.dtype

        # Ensure inputs are contiguous for simple stride arithmetic
        e = encoder_hidden_states.contiguous()
        i = hidden_states.contiguous()
        w = process_weight.contiguous()  # [H, H]

        # 1) Concatenate: out [B, T+I, H]
        out = torch.empty((B, T + I, H), dtype=dtype, device=device)

        BLOCK_l = 128
        BLOCK_h = 128
        grid_concat = (B, triton.cdiv(T + I, BLOCK_l), triton.cdiv(H, BLOCK_h))
        concat_encoder_image_kernel[grid_concat](
            e, i, out,
            B, T, I, H,
            e.stride(0), e.stride(1), e.stride(2),
            i.stride(0), i.stride(1), i.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_l=BLOCK_l, BLOCK_h=BLOCK_h,
            num_warps=4, num_stages=2,
        )

        # 2) Matmul: processed = out @ w.T, out is [B, L, H], w is [H, H]
        # We need processed as [B, L, H], where L = T+I. We compute row-wise.
        L = T + I
        processed = torch.empty((B, L, H), dtype=torch.float32, device=device)  # compute in fp32 for stability

        BLOCK_k = 64
        grid_matmul = (B, L, triton.cdiv(H, BLOCK_k))
        matmul_rowwise_kernel[grid_matmul](
            out, w, processed,
            B, L, H,
            out.stride(0), out.stride(1), out.stride(2),
            w.stride(0), w.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_k=BLOCK_k,
            num_warps=4, num_stages=2,
        )

        # 3) Slice into two streams using Triton copy kernels
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=device)

        BLOCK_l_split = 128
        BLOCK_h_split = 128

        # Copy first T rows: processed[:, :T, :] -> processed_encoder
        grid_copy_encoder = (B, triton.cdiv(T, BLOCK_l_split), triton.cdiv(H, BLOCK_h_split))
        copy_rows_kernel[grid_copy_encoder](
            processed, processed_encoder,
            B, T, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            ROW_START=0, BLOCK_l=BLOCK_l_split, BLOCK_h=BLOCK_h_split,
            num_warps=4, num_stages=2,
        )

        # Copy next I rows: processed[:, T:, :] -> processed_hidden
        grid_copy_hidden = (B, triton.cdiv(I, BLOCK_l_split), triton.cdiv(H, BLOCK_h_split))
        copy_rows_kernel[grid_copy_hidden](
            processed, processed_hidden,
            B, I, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            ROW_START=T, BLOCK_l=BLOCK_l_split, BLOCK_h=BLOCK_h_split,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden