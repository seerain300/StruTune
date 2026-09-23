import torch
import triton
import triton.language as tl


@triton.jit
def concat_encoder_image_kernel(
    e_ptr,  # *T: [B, T, H]
    i_ptr,  # *T: [B, I, H]
    out_ptr,  # *T: [B, T+I, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    e_s0, e_s1, e_s2,  # strides for e_ptr
    i_s0, i_s1, i_s2,  # strides for i_ptr
    o_s0, o_s1, o_s2,  # strides for out_ptr
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # sequence index in [0, T+I)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # hidden feature index

    mask_l = l < (T + I)
    mask_h = h < H
    mask = mask_l[:, None] & mask_h[None, :]

    # Compute pointers for encoder and image parts
    # For encoder: l < T, so source row index is l
    # For image: l >= T, so source row index is l - T
    # Build 2D pointer grids [BLOCK_l, BLOCK_h]
    # We'll use masks to select per-element.
    # Create 2D index grids: index_t or index_i depend on l
    # Triton supports broadcasting: we can form a 2D pointer via broadcasting l and h
    # out[b, l, h] = e[b, l, h] if l < T else i[b, l - T, h]
    # We form two 2D pointer arrays: ptr_e and ptr_i, then select using mask and store
    # To avoid invalid pointer creation, we can compute with scalar conditions per element.

    # We need b for each program: scalar pid_b
    # For pointer arithmetic: out_ptr + b*o_s0 + l*o_s1 + h*o_s2
    # e_ptr + b*e_s0 + l*e_s1 + h*e_s2
    # i_ptr + b*i_s0 + (l - T)*i_s1 + h*i_s2

    # We'll store via conditional: compute two loads and then pick
    # Create 2D grids for e and i
    # Note: l is 1D, h is 1D -> broadcasting to [BLOCK_l, BLOCK_h] via [:, None] and [None, :]
    # ptr_e and ptr_i shapes: [BLOCK_l, BLOCK_h]
    # mask_e: l < T
    mask_e = l[:, None] < T
    mask_i = l[:, None] >= T

    # Compute pointers for e and i (broadcast l and h)
    ptr_e = e_ptr + pid_b * e_s0 + l[:, None] * e_s1 + h[None, :] * e_s2
    ptr_i = i_ptr + pid_b * i_s0 + (l[:, None] - T) * i_s1 + h[None, :] * i_s2

    # Valid masks for e and i pointers (h always valid if mask_h, l in range if mask_l)
    mask_e2d = mask_e & mask_h[None, :]
    mask_i2d = mask_i & mask_h[None, :]

    # Load values from source (masked)
    # Triton supports tl.load with mask; ensure masked loads won't OOB
    val_e = tl.load(ptr_e, mask=mask_e2d, other=0.0)
    val_i = tl.load(ptr_i, mask=mask_i2d, other=0.0)

    # Select per element: if l < T, take val_e else val_i
    val = tl.where(mask_e2d, val_e, val_i)
    # Store to out
    out_ptr_2d = out_ptr + pid_b * o_s0 + l[:, None] * o_s1 + h[None, :] * o_s2
    tl.store(out_ptr_2d, val, mask=mask)


@triton.jit
def matmul_kernel(
    A_ptr,  # *T: [M, K] where M = B*(T+I), K = H
    W_ptr,  # *T: [K, N] where N = H (process_weight.T)
    C_ptr,  # *T: [M, N] (output)
    M: tl.int32, K: tl.int32, N: tl.int32,
    A_s0, A_s1,  # strides for A_ptr
    W_s0, W_s1,  # strides for W_ptr
    C_s0, C_s1,  # strides for C_ptr
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K]
        A_tile = tl.load(
            A_ptr + m[:, None] * A_s0 + kk[None, :] * A_s1,
            mask=(m[:, None] < M) & (kk[None, :] < K),
            other=0.0,
        )

        # W^T tile: [BLOCK_K, BLOCK_N]
        W_tile = tl.load(
            W_ptr + kk[:, None] * W_s0 + n[None, :] * W_s1,
            mask=(kk[:, None] < K) & (n[None, :] < N),
            other=0.0,
        )

        acc += tl.dot(A_tile, W_tile)

    # Store result to C
    C_ptrs = C_ptr + m[:, None] * C_s0 + n[None, :] * C_s1
    tl.store(C_ptrs, acc, mask=(m[:, None] < M) & (n[None, :] < N))


@triton.jit
def copy_rows_kernel(
    in_ptr,  # *T: [B, L, H] source
    out_ptr, # *T: [B, L_out, H] destination
    B: tl.int32, L: tl.int32, H: tl.int32, ROW_START: tl.int32,
    in_s0, in_s1, in_s2, out_s0, out_s1, out_s2,
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    r = ROW_START + pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # row indices in source
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)

    mask_r = r < (ROW_START + L)
    mask_h = h < H
    mask = (r[:, None] < (ROW_START + L)) & (h[None, :] < H)

    in_ptrs = in_ptr + pid_b * in_s0 + r[:, None] * in_s1 + h[None, :] * in_s2
    out_ptrs = out_ptr + pid_b * out_s0 + (r[:, None] - ROW_START) * out_s1 + h[None, :] * out_s2

    vals = tl.load(in_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of:
          processed = cat([encoder_hidden_states, hidden_states], dim=1) @ process_weight.T
          processed_encoder = processed[:, :text_seq_len, :]
          processed_hidden = processed[:, text_seq_len:, :]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Triton kernels require CUDA tensors."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        L = T + I

        # 1) Concatenate encoder_hidden_states and hidden_states along sequence dimension
        # out [B, L, H]
        out = torch.empty((B, L, H), dtype=hidden_states.dtype, device=hidden_states.device)
        e = encoder_hidden_states
        i = hidden_states

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

        # 2) Matmul: C = out @ process_weight.T
        # out is [B, L, H]; W is [H, H]; result C is [B, L, H]
        W_T = process_weight.transpose(0, 1).contiguous()  # [H, H]
        C = torch.empty((B, L, H), dtype=out.dtype, device=out.device)

        M = B * L
        K = H
        N = H

        grid_matmul = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        matmul_kernel[grid_matmul](
            out, W_T, C,
            M, K, N,
            out.stride(0), out.stride(1),   # A has shape [M, K], here M=L rows collapsed with batch, but we flatten: we need A as [B*L, H]. Use out contiguous or pass correctly.
            W_T.stride(0), W_T.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        # 3) Split into two outputs
        processed_encoder = torch.empty((B, T, H), dtype=C.dtype, device=C.device)
        processed_hidden = torch.empty((B, I, H), dtype=C.dtype, device=C.device)

        grid_copy_encoder = (B, triton.cdiv(T, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_encoder](
            C, processed_encoder,
            B, T, H,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            ROW_START=0, BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        grid_copy_hidden = (B, triton.cdiv(I, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_hidden](
            C, processed_hidden,
            B, I, H,
            C.stride(0), C.stride(1), C.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            ROW_START=T, BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
