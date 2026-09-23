import torch
import triton
import triton.language as tl


@triton.jit
def concat_encoder_image_kernel(
    e_ptr,        # *T: [B, T, H]
    i_ptr,        # *T: [B, I, H]
    out_ptr,      # *T: [B, T+I, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    e_s0, e_s1, e_s2,     # strides for e_ptr
    i_s0, i_s1, i_s2,     # strides for i_ptr
    o_s0, o_s1, o_s2,     # strides for out_ptr
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    # tile indices
    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)

    # masks
    mask_l = l < (T + I)
    mask_h = h < H

    # compute base pointers
    e_base = e_ptr + pid_b * e_s0
    i_base = i_ptr + pid_b * i_s0
    o_base = out_ptr + pid_b * o_s0

    # load from encoder or image depending on l
    mask_e = mask_l & (l < T)  # only valid for l < T
    mask_i = mask_l & (l >= T)  # only valid for l >= T

    # build 2D pointers for loads (broadcast h)
    e_ptrs = e_base + l[:, None] * e_s1 + h[None, :] * e_s2
    i_ptrs = i_base + (l[:, None] - T) * i_s1 + h[None, :] * i_s2

    e_vals = tl.load(e_ptrs, mask=mask_e[:, None] & mask_h[None, :], other=0.0)
    i_vals = tl.load(i_ptrs, mask=mask_i[:, None] & mask_h[None, :], other=0.0)

    out_vals = tl.where(l[:, None] < T, e_vals, i_vals)

    out_ptrs = o_base + l[:, None] * o_s1 + h[None, :] * o_s2
    tl.store(out_ptrs, out_vals, mask=mask_l[:, None] & mask_h[None, :])


@triton.jit
def matmul_kernel(
    A_ptr,        # *T: [B, L, H], input concatenated
    W_ptr,        # *T: [H, H], weight transposed
    C_ptr,        # *float32: [B, L, H], output
    B: tl.int32, L: tl.int32, H: tl.int32,
    A_s0, A_s1, A_s2,
    W_s0, W_s1, W_s2,
    C_s0, C_s1, C_s2,
    BLOCK_m: tl.constexpr, BLOCK_n: tl.constexpr, BLOCK_k: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m = pid_m * BLOCK_m + tl.arange(0, BLOCK_m)  # over rows (B*L)
    n = pid_n * BLOCK_n + tl.arange(0, BLOCK_n)  # over columns (H)
    k = tl.arange(0, BLOCK_k)                    # reduction dim (H)

    mask_m = m < (B * L)
    mask_n = n < H

    # Compute row b and local row index within L
    b = m // L
    local_m = m % L

    # Initialize accumulator
    acc = tl.zeros((BLOCK_m, BLOCK_n), dtype=tl.float32)

    # Loop over K
    for kk in range(0, H, BLOCK_k):
        k_idx = kk + k  # [BLOCK_k]
        mask_k = k_idx < H

        # Load A tile: A[b, local_m, k_idx] -> shape (BLOCK_m, BLOCK_k)
        A_ptrs = A_ptr + b[:, None] * A_s0 + local_m[:, None] * A_s1 + k_idx[None, :] * A_s2
        A_mask = mask_m[:, None] & mask_k[None, :]
        A_vals = tl.load(A_ptrs, mask=A_mask, other=0.0)  # dtype follows A_ptr

        # Load W^T tile: W^T[k_idx, n] = W[n, k_idx] -> shape (BLOCK_k, BLOCK_n)
        W_ptrs = W_ptr + n[None, :] * W_s0 + k_idx[:, None] * W_s1
        W_mask = mask_n[None, :] & mask_k[:, None]
        W_vals = tl.load(W_ptrs, mask=W_mask, other=0.0)  # dtype follows W_ptr

        # Accumulate: (BLOCK_m, BLOCK_k) @ (BLOCK_k, BLOCK_n) -> (BLOCK_m, BLOCK_n)
        acc += tl.dot(A_vals.to(tl.float32), W_vals.to(tl.float32))

    # Store result to C[b, local_m, n]
    C_ptrs = C_ptr + b[:, None] * C_s0 + local_m[:, None] * C_s1 + n[None, :] * C_s2
    tl.store(C_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def copy_rows_kernel(
    src_ptr,      # *T: [B, L, H]
    dst_ptr,      # *T: [B, L, H]
    B: tl.int32, L: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,
    dst_s0, dst_s1, dst_s2,
    ROW_START: tl.int32,
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    l = ROW_START + pid_l * BLOCK_l + tl.arange(0, BLOCK_l)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)

    mask_l = l < (L - ROW_START)
    mask_h = h < H

    src_base = src_ptr + pid_b * src_s0
    dst_base = dst_ptr + pid_b * dst_s0

    src_ptrs = src_base + l[:, None] * src_s1 + h[None, :] * src_s2
    dst_ptrs = dst_base + l[:, None] * dst_s1 + h[None, :] * dst_s2

    vals = tl.load(src_ptrs, mask=mask_l[:, None] & mask_h[None, :], other=0.0)
    tl.store(dst_ptrs, vals, mask=mask_l[:, None] & mask_h[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton implementation:
          - Concatenate encoder_hidden_states and hidden_states along sequence dimension.
          - Apply linear projection: concatenated @ process_weight.T
          - Split back into separate encoder and image streams.
        """
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]

        # Ensure contiguous for simple stride math
        e = encoder_hidden_states.contiguous()
        i = hidden_states.contiguous()
        W = process_weight.contiguous()  # [H, H]

        # 1) Concatenate into out [B, T+I, H]
        out = torch.empty((B, T + I, H), dtype=e.dtype, device=e.device)

        grid_concat = (B, triton.cdiv(T + I, 64), triton.cdiv(H, 64))
        concat_encoder_image_kernel[grid_concat](
            e, i, out,
            B, T, I, H,
            e.stride(0), e.stride(1), e.stride(2),
            i.stride(0), i.stride(1), i.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # 2) Matmul: processed = out @ W.T, result [B, T+I, H] in float32
        processed = torch.empty((B, T + I, H), dtype=torch.float32, device=e.device)

        grid_matmul = (B, triton.cdiv(T + I, 64), triton.cdiv(H, 64))
        matmul_kernel[grid_matmul](
            out, W, processed,
            B, T + I, H,
            out.stride(0), out.stride(1), out.stride(2),
            W.stride(0), W.stride(1), W.stride(2),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_m=64, BLOCK_n=64, BLOCK_k=64,
            num_warps=4, num_stages=2,
        )

        # 3) Split into two streams using Triton copy kernels
        processed_encoder = torch.empty((B, T, H), dtype=processed.dtype, device=e.device)
        processed_hidden = torch.empty((B, I, H), dtype=processed.dtype, device=e.device)

        grid_copy_encoder = (B, triton.cdiv(T, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_encoder](
            processed, processed_encoder,
            B, T + I, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            ROW_START=0, BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        grid_copy_hidden = (B, triton.cdiv(I, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_hidden](
            processed, processed_hidden,
            B, T + I, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            ROW_START=T, BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
