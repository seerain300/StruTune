import torch
import triton
import triton.language as tl


@triton.jit
def concat_encoder_image_kernel(
    e_ptr,  # *T: [B, T, H]
    i_ptr,  # *T: [B, I, H]
    out_ptr,  # *T: [B, L, H], L = T + I
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    e_s0, e_s1, e_s2,   # strides for e_ptr
    i_s0, i_s1, i_s2,   # strides for i_ptr
    o_s0, o_s1, o_s2,   # strides for out_ptr
    BLOCK_L: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # 3D grid: (B, tiles over L, tiles over H)
    pid_b = tl.program_id(0)
    pid_L = tl.program_id(1)
    pid_H = tl.program_id(2)

    l = pid_L * BLOCK_L + tl.arange(0, BLOCK_L)  # sequence index
    h = pid_H * BLOCK_H + tl.arange(0, BLOCK_H)  # hidden dim index

    # Mask for valid output indices
    mask_lh = (l[:, None] < (T + I)) & (h[None, :] < H)

    # Compute source pointers: if l < T -> encoder, else -> image at l - T
    mask_encoder = l < T
    mask_image = l >= T

    # Build per-element masks for loads
    mask_e = mask_lh & mask_encoder
    mask_i = mask_lh & mask_image

    # Pointers for encoder and image sources
    e_offsets = pid_b * e_s0 + l[:, None] * e_s1 + h[None, :] * e_s2
    i_offsets = pid_b * i_s0 + (l[:, None] - T) * i_s1 + h[None, :] * i_s2

    # Load values with masks. For masked-out, use 0.
    vals_e = tl.load(e_ptr + e_offsets, mask=mask_e, other=0)
    vals_i = tl.load(i_ptr + i_offsets, mask=mask_i, other=0)
    # Select source based on mask: if l < T use encoder, else use image
    out_vals = tl.where(mask_encoder[:, None], vals_e, tl.zeros_like(vals_e))
    # For lanes where mask_image is True, override with image values
    out_vals = tl.where(mask_image[:, None], vals_i, out_vals)

    # Store to out
    out_offsets = pid_b * o_s0 + l[:, None] * o_s1 + h[None, :] * o_s2
    tl.store(out_ptr + out_offsets, out_vals, mask=mask_lh)


@triton.jit
def matmul_kernel(
    A_ptr,  # *T: [B, L, H], we treat as [M, K] with M=B*L, K=H
    W_ptr,  # *T: [H, H], weight matrix
    C_ptr,  # *T: [B, L, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,  # B used for batch, L=T+I
    A_s0, A_s1, A_s2,  # strides for A
    W_s0, W_s1,        # strides for W (note: W is [H, H], so we only need s0,s1 for row/col)
    C_s0, C_s1, C_s2,  # strides for C
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = B*L, N = H
    L = T + I
    M = B * L

    pid_m = tl.program_id(0)  # tiles over M
    pid_n = tl.program_id(1)  # tiles over N

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # row index in A/C
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # column index in C

    # Masks for valid indices
    mask_m = m < M
    mask_n = n < H

    # Map m to (b, l) where l in [0, L), b = m // L
    l_idx = m % L
    b_idx = m // L

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension (H)
    for k0 in range(0, H, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)  # reduction index
        mask_k = k < H

        # Load A tile: shape [BLOCK_M, BLOCK_K]
        # Address: A[b, l, k] => b*A_s0 + l*A_s1 + k*A_s2
        a_ptrs = A_ptr + b_idx[:, None] * A_s0 + l_idx[:, None] * A_s1 + k[None, :] * A_s2
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(A_ptr + a_ptrs, mask=a_mask, other=0)

        # Load W^T tile: we want W[k, n] => since W is [H, H], this is W[k, n]
        # Address: W[k, n] => k*W_s0 + n*W_s1
        w_ptrs = W_ptr + k[:, None] * W_s0 + n[None, :] * W_s1
        w_mask = mask_k[:, None] & mask_n[None, :]
        w = tl.load(W_ptr + w_ptrs, mask=w_mask, other=0)

        # Accumulate (note: cast to fp32 for stable accumulation)
        acc += tl.dot(a.to(tl.float32), w.to(tl.float32))

    # Store results to C
    # Map m back to (b, l) to compute C offsets
    c_offsets = b_idx[:, None] * C_s0 + l_idx[:, None] * C_s1 + n[None, :] * C_s2
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptr + c_offsets, acc, mask=c_mask)


@triton.jit
def copy_rows_kernel(
    src_ptr,  # *T: [B, L, H]
    dst_ptr,  # *T: [B, R, H]
    B: tl.int32, L: tl.int32, H: tl.int32, R: tl.int32,
    src_s0, src_s1, src_s2,
    dst_s0, dst_s1, dst_s2,
    ROW_START: tl.int32,
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # Copy rows [ROW_START: ROW_START + R] from src to dst
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    l = ROW_START + pid_l * BLOCK_l + tl.arange(0, BLOCK_l)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)

    mask_l = (l < (ROW_START + R)) & (l < L)
    mask_h = h < H
    mask = mask_l[:, None] & mask_h[None, :]

    src_offsets = pid_b * src_s0 + l[:, None] * src_s1 + h[None, :] * src_s2
    dst_offsets = pid_b * dst_s0 + (l[:, None] - ROW_START) * dst_s1 + h[None, :] * dst_s2

    vals = tl.load(src_ptr + src_offsets, mask=mask, other=0)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        - Concatenates encoder_hidden_states and hidden_states along the sequence dimension.
        - Applies linear projection (matmul with process_weight.T).
        - Splits back into two streams.

        All tensor operations are performed by Triton kernels. No torch operations are used
        on tensors in the host code.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All inputs must be CUDA tensors for Triton kernels."

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        L = T + I

        # 1) Concatenate via Triton
        out = torch.empty((B, L, H), device=hidden_states.device, dtype=hidden_states.dtype)
        e_strides = encoder_hidden_states.stride()
        i_strides = hidden_states.stride()
        o_strides = out.stride()

        BLOCK_L = 64
        BLOCK_H = 64
        grid_concat = (B, triton.cdiv(L, BLOCK_L), triton.cdiv(H, BLOCK_H))
        concat_encoder_image_kernel[grid_concat](
            encoder_hidden_states, hidden_states, out,
            B, T, I, H,
            e_strides[0], e_strides[1], e_strides[2],
            i_strides[0], i_strides[1], i_strides[2],
            o_strides[0], o_strides[1], o_strides[2],
            BLOCK_L=BLOCK_L, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        # 2) Matmul via Triton: processed = out @ process_weight.T
        # out: [B, L, H], process_weight: [H, H]
        processed = torch.empty((B, L, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # We treat out as [M, K] with M = B*L and K = H, and W as [K, N] with N = H.
        A = out
        W = process_weight  # [H, H]
        C = processed

        A_s0, A_s1, A_s2 = A.stride()
        W_s0, W_s1 = W.stride()  # W is 2D, only need s0, s1
        C_s0, C_s1, C_s2 = C.stride()

        # Choose tiling parameters
        BLOCK_M = 128  # tile over M = B*L
        BLOCK_N = 64   # tile over N = H
        BLOCK_K = 64   # reduction chunk over K = H

        M = B * L
        grid_matmul = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        matmul_kernel[grid_matmul](
            A, W, C,
            B, T, I, H,
            A_s0, A_s1, A_s2,
            W_s0, W_s1,
            C_s0, C_s1, C_s2,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split processed into encoder and hidden parts via Triton
        processed_encoder = torch.empty((B, T, H), device=hidden_states.device, dtype=hidden_states.dtype)
        processed_hidden = torch.empty((B, I, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Copy first T rows
        grid_copy_encoder = (B, triton.cdiv(T, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_encoder](
            processed, processed_encoder,
            B, L, H, T,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            ROW_START=0, BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # Copy next I rows
        grid_copy_hidden = (B, triton.cdiv(I, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_hidden](
            processed, processed_hidden,
            B, L, H, I,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            ROW_START=T, BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden