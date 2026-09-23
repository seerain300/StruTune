import torch
import triton
import triton.language as tl


@triton.jit
def concat_encoder_image_kernel(
    e_ptr,        # *T: [B, T, H]
    i_ptr,        # *T: [B, I, H]
    out_ptr,      # *T: [B, T+I, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    e_s0, e_s1, e_s2,  # strides for e_ptr
    i_s0, i_s1, i_s2,  # strides for i_ptr
    o_s0, o_s1, o_s2,  # strides for out_ptr
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # sequence indices [0, T+I)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # hidden dim indices [0, H)

    L = T + I

    # 2D grid: (B, tiles over L, tiles over H)
    # Mask for valid b
    mask_b = pid_b < B
    # Broadcast masks for l and h
    mask_l = l < L
    mask_h = h < H
    # Compose mask for the tile
    mask = (mask_b[:, None] & mask_l[None, :] & mask_h[None, :])

    # Compute pointers
    # For encoder: condition l < T, else l - T for image
    l_is_encoder = l < T
    l_img = l - T

    # Build pointer grids for loads
    e_offsets = pid_b * e_s0 + l[None, :] * e_s1 + h[:, None] * e_s2
    i_offsets = pid_b * i_s0 + l_img[None, :] * i_s1 + h[:, None] * i_s2
    out_offsets = pid_b * o_s0 + l[None, :] * o_s1 + h[:, None] * o_s2

    # Load from encoder or image based on mask
    # Triton doesn't support masked per-element selection in pointer, so compute a combined pointer
    # by selecting between e_offsets and i_offsets using l_is_encoder. We can use where to choose.
    # However, Triton pointer arithmetic doesn't support where; instead we load with masks:
    # We'll construct two potential loads and choose via tl.where on the mask, but Triton doesn't
    # support conditional pointer loads directly. Instead, we load both with masks and then select
    # via tl.where. In practice, we perform a single load by computing pointer for each branch
    # and then using a mask. Since Triton doesn't allow dynamic branching on pointer, we implement
    # two masks and load accordingly by computing which pointer is valid per element.
    # To keep it simple and safe, we use one pointer set for each branch and mask loads:
    # We'll create a single load by constructing the pointer for the selected branch using arithmetic.

    # Construct selected offsets by combining masks:
    # When l_is_encoder, we use e_offsets; otherwise i_offsets. Triton requires static pointer grids.
    # We'll implement by computing a "selected_offsets" via tl.where(l_is_encoder, e_offsets, i_offsets)
    # Note: Triton allows elementwise operations on pointers; tl.where on pointer grid works.
    selected_offsets = tl.where(l_is_encoder[None, :], e_offsets, i_offsets)

    # Load values (masked). Default other=0.0 ensures masked positions are zero.
    vals = tl.load(out_ptr + selected_offsets, mask=mask, other=0.0)

    # Store to output
    tl.store(out_ptr + out_offsets, vals, mask=mask)


@triton.jit
def matmul_kernel(
    A_ptr, W_ptr, C_ptr,
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    A_s0, A_s1, A_s2,    # strides for A [B, T+I, H]
    W_s0, W_s1, W_s2,    # strides for W [H, H]
    C_s0, C_s1, C_s2,    # strides for C [B, T+I, H]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Flatten (B, T+I) into M rows
    M = B * (T + I)
    N = H
    K = H

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows: [0, M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # columns: [0, N)

    # Masks for valid rows/cols
    mask_m = m < M
    mask_n = n < N

    # Compute b and l from flattened m
    b = m // (T + I)
    l = m % (T + I)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)  # [0, BLOCK_K)
        mask_k = k < K

        # Load A tile: A[b, l, k] -> pointer arithmetic with strides
        # A has shape [B, T+I, H] with strides (A_s0, A_s1, A_s2)
        a_ptrs = A_ptr + b[:, None] * A_s0 + l[:, None] * A_s1 + k[None, :] * A_s2
        a_mask = (mask_m[:, None] & mask_k[None, :])
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load W^T tile: we want W[k, n], since W is [H, H]
        w_ptrs = W_ptr + k[:, None] * W_s0 + n[None, :] * W_s1
        w_mask = (mask_k[:, None] & mask_n[None, :])
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, w)

    # Write back C: C[b, l, n] using strides
    c_ptrs = C_ptr + b[:, None] * C_s0 + l[:, None] * C_s1 + n[None, :] * C_s2
    c_mask = (mask_m[:, None] & mask_n[None, :])
    # Cast back to original dtype if needed. Here we assume float32; if inputs are half, you can cast.
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def copy_rows_kernel(
    src_ptr, dest_ptr,
    B: tl.int32, NUM_ROWS: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,
    dest_s0, dest_s1, dest_s2,
    ROW_START: tl.int32,
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    l = ROW_START + pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # rows to copy
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)             # hidden dim indices

    mask_b = pid_b < B
    mask_l = l < (ROW_START + NUM_ROWS)
    mask_h = h < H
    mask = (mask_b[:, None] & mask_l[None, :] & mask_h[None, :])

    # Compute source and destination offsets
    src_offsets = pid_b * src_s0 + l[None, :] * src_s1 + h[:, None] * src_s2
    dest_offsets = pid_b * dest_s0 + l[None, :] * dest_s1 + h[:, None] * dest_s2

    vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(dest_ptr + dest_offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]

        # Ensure tensors are contiguous and on same device/dtype
        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1) Concatenate using Triton
        concatenated = torch.empty((B, T + I, H), device=device, dtype=dtype)

        # Strides
        e_s0, e_s1, e_s2 = encoder_hidden_states.stride()
        i_s0, i_s1, i_s2 = hidden_states.stride()
        o_s0, o_s1, o_s2 = concatenated.stride()

        grid_concat = (B, triton.cdiv(T + I, 64), triton.cdiv(H, 64))
        concat_encoder_image_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, T, I, H, e_s0, e_s1, e_s2, i_s0, i_s1, i_s2, o_s0, o_s1, o_s2,
            BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # 2) Matmul: concatenated @ process_weight.T
        # process_weight: [H, H], output processed: [B, T+I, H]
        processed = torch.empty((B, T + I, H), device=device, dtype=torch.float32)  # compute in float32 for stability

        W = process_weight  # [H, H]
        A = concatenated    # [B, T+I, H]

        A_s0, A_s1, A_s2 = A.stride()
        W_s0, W_s1, W_s2 = W.stride()
        C_s0, C_s1, C_s2 = processed.stride()

        # Choose block sizes; simple robust choice
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 64

        grid_matmul = (B * (T + I), triton.cdiv(H, BLOCK_N))
        matmul_kernel[grid_matmul](
            A, W, processed,
            B, T, I, H,
            A_s0, A_s1, A_s2,
            W_s0, W_s1, W_s2,
            C_s0, C_s1, C_s2,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split using Triton copy kernels
        processed_encoder = torch.empty((B, T, H), device=device, dtype=processed.dtype)
        processed_hidden = torch.empty((B, I, H), device=device, dtype=processed.dtype)

        # Strides
        p_s0, p_s1, p_s2 = processed.stride()
        pe_s0, pe_s1, pe_s2 = processed_encoder.stride()
        ph_s0, ph_s1, ph_s2 = processed_hidden.stride()

        # Copy first T rows
        grid_copy_encoder = (B, triton.cdiv(T, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_encoder](
            processed, processed_encoder,
            B, T, H,
            p_s0, p_s1, p_s2,
            pe_s0, pe_s1, pe_s2,
            ROW_START=0, BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # Copy next I rows
        grid_copy_hidden = (B, triton.cdiv(I, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_hidden](
            processed, processed_hidden,
            B, I, H,
            p_s0, p_s1, p_s2,
            ph_s0, ph_s1, ph_s2,
            ROW_START=T, BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # Match original dtype (the original outputs are float32; adjust if needed)
        processed_encoder = processed_encoder.to(dtype)
        processed_hidden = processed_hidden.to(dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
