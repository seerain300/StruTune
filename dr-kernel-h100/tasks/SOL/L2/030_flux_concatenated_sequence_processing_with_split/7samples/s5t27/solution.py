import torch
import triton
import triton.language as tl


@triton.jit
def concat_encoder_image_kernel(
    e_ptr,                # *const T: [B, T, H]
    i_ptr,                # *const T: [B, I, H]
    out_ptr,              # *T: [B, T+I, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    e_s0, e_s1, e_s2,     # strides for e_ptr (batch, seq, hidden)
    i_s0, i_s1, i_s2,     # strides for i_ptr
    o_s0, o_s1, o_s2,     # strides for out_ptr
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # 3D launch: (B, tiles over L=T+I, tiles over H)
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    b = pid_b
    l_offsets = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # [BLOCK_l]
    h_offsets = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # [BLOCK_h]

    # Create 2D grid of indices (l, h) for the tile
    L = T + I
    # Broadcast to 2D
    l = l_offsets[:, None]  # [BLOCK_l, 1]
    h = h_offsets[None, :]  # [1, BLOCK_h]

    # Valid mask
    mask = (l < L) & (h < H) & (b < B)

    # Compute source indices: if l < T -> from e_ptr; else -> from i_ptr at index l - T
    is_encoder = l < T
    src_seq = tl.where(is_encoder, l, l - T)

    # Compute addresses
    # e_ptr[b, src_seq, h] and i_ptr[b, l - T, h]
    addr_e = e_ptr + b * e_s0 + src_seq * e_s1 + h * e_s2
    addr_i = i_ptr + b * i_s0 + (l - T) * i_s1 + h * i_s2
    # Select based on is_encoder
    src_vals = tl.where(is_encoder, tl.load(addr_e, mask=mask, other=0.0), tl.load(addr_i, mask=mask, other=0.0))

    # Destination address for out[b, l, h]
    addr_out = out_ptr + b * o_s0 + l * o_s1 + h * o_s2
    tl.store(addr_out, src_vals, mask=mask)


@triton.jit
def matmul_kernel(
    A_ptr,  # *const T: [B, L, H], where L=T+I
    W_ptr,  # *const T: [H, H] (process_weight)
    C_ptr,  # *T: [B, L, H] output
    B: tl.int32, L: tl.int32, H: tl.int32,
    A_s0, A_s1, A_s2,     # strides for A (batch, seq, hidden)
    W_s0, W_s1,           # strides for W (hidden, hidden)
    C_s0, C_s1, C_s2,     # strides for C
    BLOCK_M: tl.constexpr,  # tile over rows (B*L)
    BLOCK_N: tl.constexpr,  # tile over columns (H)
    BLOCK_K: tl.constexpr,  # reduction tile over K (H)
):
    # Grid over tiles of (M=B*L, N=H) with a single axis for simplicity
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows in [0, B*L)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # columns in [0, H)

    # Map m_offsets to (b, l) by integer division/mod
    # b = m // L, l = m % L
    b_idx = m_offsets // L
    l_idx = m_offsets % L

    # Masks
    mask_m = (m_offsets < (B * L))[:, None]
    mask_n = (n_offsets < H)[None, :]
    mask = mask_m & mask_n

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K=H in chunks
    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = (k_offsets < H)[None, :]       # [1, BLOCK_K]

        # Load A tile: A[b, l, k] -> shape (BLOCK_M, BLOCK_K)
        # Compute addresses: A[b*b_s0 + l*a_s1 + k*a_s2]
        # Note: A_s0, A_s1, A_s2 are provided in host code.
        A_ptrs = A_ptr + b_idx[:, None] * A_s0 + l_idx[:, None] * A_s1 + k_offsets[None, :] * A_s2
        A_tile = tl.load(A_ptrs, mask=mask_m & mask_k, other=0.0)

        # Load W^T tile: W[k, n] -> shape (BLOCK_K, BLOCK_N)
        W_ptrs = W_ptr + k_offsets[:, None] * W_s0 + n_offsets[None, :] * W_s1
        W_tile = tl.load(W_ptrs, mask=mask_k & mask_n, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile.to(tl.float32), W_tile.to(tl.float32))

    # Store result into C[b, l, n] = acc
    # Compute output addresses and store
    C_ptrs = C_ptr + b_idx[:, None] * C_s0 + l_idx[:, None] * C_s1 + n_offsets[None, :] * C_s2
    tl.store(C_ptrs, acc, mask=mask)


@triton.jit
def copy_rows_kernel(
    src_ptr, dst_ptr,        # src: [B, L, H], dst: [B, ? , H]
    B: tl.int32, L: tl.int32, H: tl.int32,
    start_row: tl.int32,     # starting row in src to copy
    src_s0, src_s1, src_s2,  # strides for src
    dst_s0, dst_s1, dst_s2,  # strides for dst
    NUM_ROWS: tl.int32,      # number of rows to copy
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # Grid over (B, tiles over rows, tiles over H)
    pid_b = tl.program_id(0)
    pid_rows = tl.program_id(1)
    pid_h = tl.program_id(2)

    b = pid_b

    row_offsets = pid_rows * BLOCK_l + tl.arange(0, BLOCK_l)  # [BLOCK_l]
    h_offsets = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)      # [BLOCK_h]

    rows = start_row + row_offsets[:, None]                  # [BLOCK_l, 1]
    cols = h_offsets[None, :]                                # [1, BLOCK_h]

    mask_rows = (rows < (start_row + NUM_ROWS)) & (rows < L)
    mask_h = (cols < H)
    mask = mask_rows & mask_h

    src_addrs = src_ptr + b * src_s0 + rows * src_s1 + cols * src_s2
    vals = tl.load(src_addrs, mask=mask, other=0.0)

    dst_addrs = dst_ptr + b * dst_s0 + rows * dst_s1 + cols * dst_s2
    tl.store(dst_addrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of:
          concatenated = cat([encoder_hidden_states, hidden_states], dim=1)
          processed = concatenated @ process_weight.T
          processed_encoder = processed[:, :T, :]
          processed_hidden = processed[:, T:, :]
        All operations are performed by Triton kernels; no torch ops on tensors in host.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All inputs must be CUDA tensors for Triton kernels."

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        L = T + I

        # Ensure contiguous for predictable strides
        e = encoder_hidden_states.contiguous()
        i = hidden_states.contiguous()
        W = process_weight.contiguous()

        # 1) Concatenate using Triton
        out = torch.empty((B, L, H), device=hidden_states.device, dtype=hidden_states.dtype)
        grid_concat = (B, triton.cdiv(L, 64), triton.cdiv(H, 64))
        concat_encoder_image_kernel[grid_concat](
            e, i, out,
            B, T, I, H,
            e.stride(0), e.stride(1), e.stride(2),
            i.stride(0), i.stride(1), i.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # 2) Matmul processed = out @ W.T using Triton
        processed = torch.empty((B, L, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # We launch a 2D grid over (M=B*L, N=H) tiles. For simplicity, choose 64x64 tiles.
        grid_m = (B * L + 64 - 1) // 64
        grid_n = (H + 64 - 1) // 64
        grid_matmul = (grid_m, grid_n)

        matmul_kernel[grid_matmul](
            out, W, processed,
            B, L, H,
            out.stride(0), out.stride(1), out.stride(2),
            W.stride(0), W.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2,
        )

        # 3) Copy rows for split using Triton
        processed_encoder = torch.empty((B, T, H), device=hidden_states.device, dtype=hidden_states.dtype)
        processed_hidden = torch.empty((B, I, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # First T rows
        grid_copy_encoder = (B, triton.cdiv(T, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_encoder](
            processed, processed_encoder,
            B, L, H,
            0,                           # start_row
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            T,                           # NUM_ROWS = T
            BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # Next I rows
        grid_copy_hidden = (B, triton.cdiv(I, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_hidden](
            processed, processed_hidden,
            B, L, H,
            T,                           # start_row
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            I,                           # NUM_ROWS = I
            BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
