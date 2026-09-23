import torch
import triton
import triton.language as tl


@triton.jit
def concatenate_sequences_kernel(
    e_ptr, i_ptr, out_ptr,
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    e_s0, e_s1, e_s2,
    i_s0, i_s1, i_s2,
    o_s0, o_s1, o_s2,
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # 3D grid: (batch, tiles over L=T+I, tiles over H)
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    L = T + I
    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)          # sequence indices
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)          # hidden dim indices

    # Create 2D mesh for the tile
    L_broadcast = l[:, None]  # shape [BLOCK_l, 1]
    H_broadcast = h[None, :]  # shape [1, BLOCK_h]

    # Mask to stay in bounds
    mask_l = L_broadcast < L
    mask_h = H_broadcast < H
    mask = mask_l & mask_h  # shape [BLOCK_l, BLOCK_h]

    # Compute base offsets
    e_offset = pid_b * e_s0 + L_broadcast * e_s1 + H_broadcast * e_s2
    i_offset = pid_b * i_s0 + (L_broadcast - T) * i_s1 + H_broadcast * i_s2
    o_offset = pid_b * o_s0 + L_broadcast * o_s1 + H_broadcast * o_s2

    # Select source: if l < T, use encoder_hidden_states; else, use hidden_states
    use_encoder = L_broadcast < T
    # For elements where l >= T, mask the encoder loads; for elements where l < T, mask the image loads.
    mask_encoder = mask & use_encoder
    mask_image = mask & (~use_encoder)

    # Load from encoder or image
    # For masked-out loads, use other=0 to avoid reading invalid memory.
    e_val = tl.load(e_ptr + e_offset, mask=mask_encoder, other=0.0)
    i_val = tl.load(i_ptr + i_offset, mask=mask_image, other=0.0)
    out_val = tl.where(use_encoder, e_val, i_val)

    # Store
    tl.store(out_ptr + o_offset, out_val, mask=mask)


@triton.jit
def matmul_kernel(
    A_ptr, W_ptr, C_ptr,
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    A_s0, A_s1, A_s2,
    W_s0, W_s1, W_s2,
    C_s0, C_s1, C_s2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Batched matmul: compute C[b, m, n] = sum_k A[b, m, k] * W[k, n]
    # Flatten M = B*(T+I), N = H, K = H
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = B * (T + I)
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # row indices in A/C
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # column indices in C

    # Masks
    mask_m = m < M
    mask_n = n < H
    mask_m_broadcast = mask_m[:, None]            # [BLOCK_M, 1]
    mask_n_broadcast = mask_n[None, :]            # [1, BLOCK_N]

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, H, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)  # reduction indices

        # Compute offsets for A[b, m, k]
        # m maps to (b, l) via b = m // (T+I), l = m % (T+I)
        l = m % (T + I)
        b = m // (T + I)

        # A[b, l, k] addressing
        A_offset = b[:, None] * A_s0 + l[:, None] * A_s1 + k[None, :] * A_s2  # shape [BLOCK_M, BLOCK_K]

        # W[k, n] addressing (W is [H, H], we pass W as [H, H])
        W_offset = k[:, None] * W_s0 + n[None, :] * W_s1  # shape [BLOCK_K, BLOCK_N]

        # Masks for loads
        mask_A = mask_m_broadcast & (k[None, :] < H)
        mask_W = (k[:, None] < H) & mask_n_broadcast

        # Load tiles
        A_tile = tl.load(A_ptr + A_offset, mask=mask_A, other=0.0)  # [BLOCK_M, BLOCK_K]
        W_tile = tl.load(W_ptr + W_offset, mask=mask_W, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(A_tile, W_tile)

    # Store results to C[b, m, n] where m = b*(T+I) + l
    # We need to recover (b, l) from flattened m
    b_out = m // (T + I)
    l_out = m % (T + I)
    C_offset = b_out[:, None] * C_s0 + l_out[:, None] * C_s1 + n[None, :] * C_s2  # [BLOCK_M, BLOCK_N]
    store_mask = mask_m_broadcast & mask_n_broadcast
    tl.store(C_ptr + C_offset, acc, mask=store_mask)


@triton.jit
def copy_rows_kernel(
    src_ptr, dst_ptr,
    B: tl.int32, NUM_ROWS: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,
    dst_s0, dst_s1, dst_s2,
    ROW_START: tl.int32,
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # 3D grid: (batch, tiles over rows, tiles over H)
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    rows = ROW_START + pid_l * BLOCK_l + tl.arange(0, BLOCK_l)          # row indices in src/dst
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)                          # hidden dim indices

    mask_rows = rows < (B * NUM_ROWS)  # but NUM_ROWS is number of rows to copy, not total
    # Since we launch with grid considering B, we should restrict to B. Instead, we simply assume grid uses B and NUM_ROWS
    mask_rows = rows < NUM_ROWS
    mask_h = h < H
    mask = mask_rows[:, None] & mask_h[None, :]

    # src offsets: src[b, rows, h]
    src_offset = pid_b * src_s0 + rows[:, None] * src_s1 + h[None, :] * src_s2
    # dst offsets: dst[b, rows - ROW_START, h]
    dst_offset = pid_b * dst_s0 + (rows[:, None] - ROW_START) * dst_s1 + h[None, :] * dst_s2

    # Load and store
    vals = tl.load(src_ptr + src_offset, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offset, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Shapes
        B, T, H = encoder_hidden_states.shape
        _, I, H2 = hidden_states.shape
        assert H == H2, "hidden_dim must match"
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Ensure process_weight is [H, H] and on device/dtype
        W = process_weight  # [H, H]
        assert W.shape[0] == H and W.shape[1] == H, "process_weight must be [hidden_dim, hidden_dim]"

        # 1) Concatenate along sequence dimension: [B, T+I, H]
        L = T + I
        out = torch.empty((B, L, H), dtype=dtype, device=device)
        grid_concat = (B, triton.cdiv(L, 64), triton.cdiv(H, 64))
        concatenate_sequences_kernel[grid_concat](
            encoder_hidden_states, hidden_states, out,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # 2) Matmul: processed = out @ W^T
        processed = torch.empty((B, L, H), dtype=dtype, device=device)
        grid_matmul = (triton.cdiv(B * L, 64), triton.cdiv(H, 64))
        # Note: We pass W as [H, H] which is process_weight.T in original (process_weight is [H, H])
        matmul_kernel[grid_matmul](
            out, W, processed,
            B, T, I, H,
            out.stride(0), out.stride(1), out.stride(2),
            W.stride(0), W.stride(1), W.stride(2),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2,
        )

        # 3) Split into two streams
        processed_encoder = torch.empty((B, T, H), dtype=dtype, device=device)
        grid_first = (B, triton.cdiv(T, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_first](
            processed, processed_encoder,
            B, T, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            ROW_START=0, BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        processed_hidden = torch.empty((B, I, H), dtype=dtype, device=device)
        grid_second = (B, triton.cdiv(I, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_second](
            processed, processed_hidden,
            B, I, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            ROW_START=T, BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
