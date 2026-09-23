import torch
import triton
import triton.language as tl


@triton.jit
def copy_rows_kernel(
    in_ptr, out_ptr,
    B, L, H,
    in_s0, in_s1, in_s2,
    out_s0, out_s1, out_s2,
    ROW_START: tl.constexpr,  # starting row in input to copy
    BLOCK_l: tl.constexpr,    # tile size along row dimension
    BLOCK_h: tl.constexpr,    # tile size along hidden dimension
):
    # program ids: we tile over (batch, rows, hidden)
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    # compute indices for this program
    l = ROW_START + pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # source row indices
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)              # hidden indices

    # masks for bounds
    mask_b = pid_b < B
    mask_l = l < L
    mask_h = h < H
    # 2D mask for the tile
    mask = mask_b & mask_l[:, None] & mask_h[None, :]

    # compute input and output pointers for the tile
    # in: [B, L, H] -> in_ptr + b*in_s0 + l*in_s1 + h*in_s2
    in_offsets = pid_b * in_s0 + l[:, None] * in_s1 + h[None, :] * in_s2
    # out: [B, L, H] but we are copying to out at row indices 'l' (already adjusted by ROW_START)
    out_rows = l - ROW_START  # if ROW_START is 0, this is just l
    out_offsets = pid_b * out_s0 + out_rows[:, None] * out_s1 + h[None, :] * out_s2

    # load and store
    vals = tl.load(in_ptr + in_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + out_offsets, vals, mask=mask)


@triton.jit
def matmul_kernel(
    A_ptr, W_ptr, C_ptr,
    B, L, H,
    A_s0, A_s1, A_s2,
    W_s0, W_s1, W_s2,
    C_s0, C_s1, C_s2,
    BLOCK_M: tl.constexpr,  # tile size for rows M = B*L
    BLOCK_N: tl.constexpr,  # tile size for hidden dim (columns)
    BLOCK_K: tl.constexpr,  # reduction tile size for hidden dim
):
    # We tile over M (rows) and N (columns), reduce over K (hidden dim).
    pid_m = tl.program_id(0)  # over tiles of M
    pid_n = tl.program_id(1)  # over tiles of N

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # row indices in [0, B*L)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # hidden column indices

    # Masks for bounds
    mask_m = m < (B * L)
    mask_n = n < H
    # Compute batch and sequence indices for each row m
    b = m // L
    l = m % L

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, H, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)  # reduction indices
        mask_k = k < H

        # Load A tile: [BLOCK_M, BLOCK_K]
        # A[b, l, k] -> offset = b*A_s0 + l*A_s1 + k*A_s2
        A_offsets = b[:, None] * A_s0 + l[:, None] * A_s1 + k[None, :] * A_s2
        A_mask = (b[:, None] < B) & (l[:, None] < L) & (mask_k[None, :])
        A_tile = tl.load(A_ptr + A_offsets, mask=A_mask, other=0.0)

        # Load W^T tile: we want W[k, n] -> offset = k*W_s0 + n*W_s1
        W_offsets = k[:, None] * W_s0 + n[None, :] * W_s1
        W_mask = (mask_k[:, None]) & (mask_n[None, :])
        Wt_tile = tl.load(W_ptr + W_offsets, mask=W_mask, other=0.0)

        # Accumulate
        # Note: tl.dot expects last dimension of first operand = rows of second operand.
        # Here A_tile: (M, K), Wt_tile: (K, N) -> (M, N)
        acc += tl.dot(A_tile.to(tl.float32), Wt_tile.to(tl.float32))

    # Store result to C[b, l, n] = C[ m, n ], m = b*L + l
    C_offsets = m[:, None] * C_s0 + l[None, :] * C_s1 + n[None, :] * C_s2
    C_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptr + C_offsets, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, H]
        encoder_hidden_states: [B, T, H]
        process_weight: [H, H]
        returns (processed_encoder: [B, T, H], processed_hidden: [B, I, H])
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "ModelNew requires CUDA tensors for Triton kernels."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        L = T + I

        # 1) Concatenate along sequence dimension using Triton (two copy kernels)
        out = torch.empty((B, L, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Kernel 1: copy encoder rows
        BLOCK_l_e = 64
        BLOCK_h = 64
        grid_encoder = (B, T, triton.cdiv(H, BLOCK_h))
        copy_rows_kernel[grid_encoder](
            encoder_hidden_states, out,
            B, T, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            ROW_START=0,
            BLOCK_l=BLOCK_l_e, BLOCK_h=BLOCK_h,
            num_warps=4, num_stages=2,
        )

        # Kernel 2: copy image rows starting at row T
        grid_image = (B, I, triton.cdiv(H, BLOCK_h))
        copy_rows_kernel[grid_image](
            hidden_states, out,
            B, I, H,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            ROW_START=T,
            BLOCK_l=BLOCK_l_e, BLOCK_h=BLOCK_h,
            num_warps=4, num_stages=2,
        )

        # 2) Matmul: processed = out @ process_weight.T
        processed = torch.empty((B, L, H), device=hidden_states.device, dtype=hidden_states.dtype)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (triton.cdiv(B * L, BLOCK_M), triton.cdiv(H, BLOCK_N))
        matmul_kernel[grid](
            out, process_weight, processed,
            B, L, H,
            out.stride(0), out.stride(1), out.stride(2),
            process_weight.stride(0), process_weight.stride(1), process_weight.stride(2),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split into two streams using Triton copy kernels
        processed_encoder = torch.empty((B, T, H), device=hidden_states.device, dtype=hidden_states.dtype)
        grid_encoder_split = (B, T, triton.cdiv(H, BLOCK_h))
        copy_rows_kernel[grid_encoder_split](
            processed, processed_encoder,
            B, T, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            ROW_START=0,
            BLOCK_l=BLOCK_l_e, BLOCK_h=BLOCK_h,
            num_warps=4, num_stages=2,
        )

        processed_hidden = torch.empty((B, I, H), device=hidden_states.device, dtype=hidden_states.dtype)
        grid_image_split = (B, I, triton.cdiv(H, BLOCK_h))
        copy_rows_kernel[grid_image_split](
            processed, processed_hidden,
            B, I, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            ROW_START=T,
            BLOCK_l=BLOCK_l_e, BLOCK_h=BLOCK_h,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden