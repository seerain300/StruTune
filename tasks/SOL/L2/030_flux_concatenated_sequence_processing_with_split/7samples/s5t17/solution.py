import torch
import triton
import triton.language as tl


@triton.jit
def concat_encoder_image_kernel(
    e_ptr, i_ptr, out_ptr,
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    e_s0, e_s1, e_s2,   # strides for e_ptr
    i_s0, i_s1, i_s2,   # strides for i_ptr
    o_s0, o_s1, o_s2,   # strides for out_ptr
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # 3D grid: (B, ceil((T+I)/BLOCK_l), ceil(H/BLOCK_h))
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    # sequence indices
    l_offsets = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)
    h_offsets = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)

    # broadcast to 2D tile
    L = T + I
    mask = (l_offsets[:, None] < L) & (h_offsets[None, :] < H)

    # Compute source indices for encoder and image
    mask_e = l_offsets[:, None] < T
    mask_i = (l_offsets[:, None] >= T) & (l_offsets[:, None] < L)

    # Compute per-element addresses (broadcasted)
    # e addresses: out[b, l, h] = e[b, l, h]
    e_offsets = (pid_b * e_s0) + (l_offsets[:, None] * e_s1) + (h_offsets[None, :] * e_s2)
    # i addresses: out[b, l, h] = i[b, l - T, h]
    i_offsets = (pid_b * i_s0) + ((l_offsets[:, None] - T) * i_s1) + (h_offsets[None, :] * i_s2)
    # out addresses: out[b, l, h]
    o_offsets = (pid_b * o_s0) + (l_offsets[:, None] * o_s1) + (h_offsets[None, :] * o_s2)

    # Load with masks
    e_vals = tl.load(e_ptr + e_offsets, mask=mask & mask_e, other=0.0)
    i_vals = tl.load(i_ptr + i_offsets, mask=mask & mask_i, other=0.0)

    # Compose values: for l < T take e, else take i
    out_vals = tl.where(l_offsets[:, None] < T, e_vals, i_vals)

    # Store
    tl.store(out_ptr + o_offsets, out_vals, mask=mask)


@triton.jit
def matmul_kernel(
    A_ptr, W_ptr, C_ptr,
    M: tl.int32, N: tl.int32, K: tl.int32,  # M = B*(T+I), N = H, K = H
    A_s0, A_s1, A_s2,   # strides for A: [M, N] but we pass [B, L, H] by flattening M = B*L
    W_s0, W_s1,         # strides for W: [K, N] which is process_weight.T [H, H]
    C_s0, C_s1, C_s2,   # strides for C: [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over tiles (rows, cols)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)

        # Compute A tile: A[rows, k] with shape [BLOCK_M, BLOCK_K]
        # We pass A as [M, N] and address as ((rows)*N + k)
        A_tile = tl.load(
            A_ptr + rows[:, None] * A_s0 + k[None, :] * A_s1,
            mask=(rows[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )

        # Compute W tile: W[k, cols] with shape [BLOCK_K, BLOCK_N]
        W_tile = tl.load(
            W_ptr + k[:, None] * W_s0 + cols[None, :] * W_s1,
            mask=(k[:, None] < K) & (cols[None, :] < N),
            other=0.0,
        )

        # Accumulate
        acc += tl.dot(A_tile, W_tile)

    # Store result
    C_ptrs = C_ptr + rows[:, None] * C_s0 + cols[None, :] * C_s2
    tl.store(
        C_ptrs,
        acc,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def copy_rows_kernel(
    src_ptr, dst_ptr,
    B: tl.int32, NUM_ROWS: tl.int32, N: tl.int32,  # NUM_ROWS is either T or I
    src_s0, src_s1, src_s2,
    dst_s0, dst_s1, dst_s2,
    ROW_START: tl.int32,  # starting row index in src
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # 3D grid: (B, tiles across NUM_ROWS, tiles across N)
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    l_offsets = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # row indices within the slice
    h_offsets = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # column indices

    mask_rows = (ROW_START + l_offsets) < (ROW_START + NUM_ROWS)
    mask_cols = h_offsets < N
    mask = mask_rows[:, None] & mask_cols[None, :]

    # src addresses: src[pid_b, ROW_START + l_offsets, h_offsets]
    src_offsets = (pid_b * src_s0) + ((ROW_START + l_offsets)[:, None] * src_s1) + (h_offsets[None, :] * src_s2)
    # dst addresses: dst[pid_b, l_offsets, h_offsets]
    dst_offsets = (pid_b * dst_s0) + (l_offsets[:, None] * dst_s1) + (h_offsets[None, :] * dst_s2)

    vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dim into 'concatenated'.
        2) Compute processed = concatenated @ process_weight.T using Triton matmul kernel.
        3) Split processed into processed_encoder and processed_hidden.
        All tensor ops are Triton kernels launched from forward; no torch ops are used on tensors in host.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        L = T + I

        # Allocate output for concatenation
        concatenated = torch.empty((B, L, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch concatenation kernel
        BLOCK_l = 64
        BLOCK_h = 64
        grid_concat = (
            B,
            triton.cdiv(L, BLOCK_l),
            triton.cdiv(H, BLOCK_h),
        )
        concat_encoder_image_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_l=BLOCK_l, BLOCK_h=BLOCK_h,
            num_warps=4, num_stages=2,
        )

        # Matmul: processed = concatenated @ process_weight.T
        # Ensure process_weight.T is [H, H]
        W_T = process_weight.t().contiguous()  # [H, H]
        M = B * L
        N = H
        K = H

        processed = torch.empty((B, L, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch matmul kernel: treat A as [M, N] by flattening (we pass strides accordingly)
        # A's actual layout is [B, L, H]; we address as ((rows)*N + k). We need to map rows -> (b, l).
        # Simpler: make A contiguous as [M, N] and pass its strides. Use torch.reshape with contiguous.
        # Reshape A to [M, N]
        A_2d = concatenated.reshape(M, N).contiguous()

        grid_matmul = (
            triton.cdiv(M, 64),  # rows tiles
            triton.cdiv(N, 64),  # cols tiles
        )
        matmul_kernel[grid_matmul](
            A_2d, W_T, processed,  # C is [M, N]
            M, N, K,
            A_2d.stride(0), A_2d.stride(1),
            W_T.stride(0), W_T.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        # Split into two outputs using copy kernels
        processed_encoder = torch.empty((B, T, H), device=hidden_states.device, dtype=hidden_states.dtype)
        processed_hidden = torch.empty((B, I, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Copy first T rows
        grid_copy_encoder = (B, triton.cdiv(T, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_encoder](
            processed, processed_encoder,
            B, T, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            ROW_START=0, BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # Copy next I rows
        grid_copy_hidden = (B, triton.cdiv(I, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_hidden](
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
