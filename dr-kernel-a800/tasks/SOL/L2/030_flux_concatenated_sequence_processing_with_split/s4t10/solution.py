import torch
import triton
import triton.language as tl


@triton.jit
def concat_rows_to_A_kernel(
    enc_ptr,       # [B, T, H]
    img_ptr,       # [B, I, H]
    A_ptr,         # [M, H], M = B*(T+I)
    B, T, I, H,    # dims
    stride_b_e, stride_t_e, stride_h_e,  # enc strides
    stride_b_i, stride_i_i, stride_h_i,  # img strides
    stride_b_a, stride_h_a,               # A strides
    BLOCK_H: tl.constexpr,
):
    # Each program handles a tile of columns for one output row
    row = tl.program_id(0)  # in [0, M)
    col_start = tl.program_id(1) * BLOCK_H
    offs = col_start + tl.arange(0, BLOCK_H)
    mask = offs < H

    # Compute batch and seq index from row
    total_seq = T + I
    b = row // total_seq
    seq = row % total_seq

    # Determine source tensor and row index
    is_encoder = seq < T
    if is_encoder:
        src_ptr = enc_ptr + b * stride_b_e + seq * stride_t_e + offs * stride_h_e
    else:
        src_row = seq - T
        src_ptr = img_ptr + b * stride_b_i + src_row * stride_i_i + offs * stride_h_i

    vals = tl.load(src_ptr, mask=mask, other=0.0)
    dst_ptr = A_ptr + row * stride_b_a + offs * stride_h_a
    tl.store(dst_ptr, vals, mask=mask)


@triton.jit
def triton_matmul_kernel(
    A_ptr,           # [M, H], M = B*(T+I)
    B_ptr,           # [H, H] (process_weight.T)
    C_ptr,           # [M, H] output
    B, T, I, H,      # dims
    stride_am, stride_an,   # A strides
    stride_bk, stride_bh,   # B strides
    stride_cm, stride_cn,   # C strides
    BLOCK_M: tl.constexpr,  # tile size along M
    BLOCK_N: tl.constexpr,  # tile size along N (columns)
    BLOCK_K: tl.constexpr,  # tile size along reduction dim
):
    # Program ids map to tiles in M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K (hidden_dim)
    for k_start in range(0, H, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_an
        a_mask = (offs_m[:, None] < B * (T + I)) & (offs_k[None, :] < H)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load B tile: [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bh
        b_mask = (offs_k[:, None] < H) & (offs_n[None, :] < H)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Store results to C
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < B * (T + I)) & (offs_n[None, :] < H)
    # Cast back to original dtype of A (assuming float32 for simplicity; adjust if needed)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def copy_rows_to_encoder_kernel(
    C_ptr,           # [M, H], M = B*(T+I)
    out_ptr,         # [B, T, H]
    B, T, I, H,
    stride_c0, stride_c1,
    out_stride_b, out_stride_t, out_stride_h,
    batch_id,
):
    # Each program handles a tile of columns for one output row
    row_e = tl.program_id(0)  # in [0, B*T)
    col_start = tl.program_id(1) * 64
    offs = col_start + tl.arange(0, 64)
    mask = offs < H

    # Source row in C (concatenated rows)
    src_row = row_e
    src_ptr = C_ptr + src_row * stride_c0 + offs * stride_c1

    # Destination row in encoder output for given batch
    dest_b = batch_id
    dest_t = row_e % T
    dest_ptr = out_ptr + dest_b * out_stride_b + dest_t * out_stride_t + offs * out_stride_h

    vals = tl.load(src_ptr, mask=mask, other=0.0)
    tl.store(dest_ptr, vals, mask=mask)


@triton.jit
def copy_rows_to_hidden_kernel(
    C_ptr,           # [M, H], M = B*(T+I)
    out_ptr,         # [B, I, H]
    B, T, I, H,
    stride_c0, stride_c1,
    out_stride_b, out_stride_i, out_stride_h,
    batch_id,
):
    # Each program handles a tile of columns for one output row
    row_h = tl.program_id(0)  # in [B*T, M)
    col_start = tl.program_id(1) * 64
    offs = col_start + tl.arange(0, 64)
    mask = offs < H

    # Source row in C
    src_row = row_h
    src_ptr = C_ptr + src_row * stride_c0 + offs * stride_c1

    # Destination row in hidden output for given batch
    dest_b = batch_id
    dest_i = row_h - B * T
    dest_ptr = out_ptr + dest_b * out_stride_b + dest_i * out_stride_i + offs * out_stride_h

    vals = tl.load(src_ptr, mask=mask, other=0.0)
    tl.store(dest_ptr, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation:
        - Concatenate [B, T, H] and [B, I, H] into A [B*(T+I), H] via Triton.
        - Compute C = A @ process_weight.T via Triton matmul kernel.
        - Split C into encoder and hidden outputs via Triton row-copy kernels.
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be [B, S, H]"
        assert process_weight.dim() == 2, "process_weight must be [H, H]"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H and process_weight.shape[1] == H, "Hidden dim mismatch"

        # Ensure contiguous tensors
        enc = encoder_hidden_states.contiguous()
        img = hidden_states.contiguous()
        weight_T = process_weight.t().contiguous()  # [H, H]
        device = enc.device

        # Build A: concatenated [B*(T+I), H] via Triton
        M = B * (T + I)
        A = torch.empty((M, H), dtype=enc.dtype, device=device)

        BLOCK_H = 64
        grid_concat = (M, triton.cdiv(H, BLOCK_H))
        concat_rows_to_A_kernel[grid_concat](
            enc, img, A,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            img.stride(0), img.stride(1), img.stride(2),
            A.stride(0), A.stride(1),
            BLOCK_H=BLOCK_H,
        )

        # Allocate output C and run Triton matmul
        C = torch.empty((M, H), dtype=torch.float32, device=device)  # compute in float32 for numerical stability

        # Choose tile sizes. Reasonable defaults for variety of H:
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        triton_matmul_kernel[grid](
            A, weight_T, C,
            B, T, I, H,
            A.stride(0), A.stride(1),
            weight_T.stride(0), weight_T.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Cast back to original dtype
        if enc.dtype != torch.float32:
            C = C.to(enc.dtype)

        # Split into encoder and hidden streams via Triton row-copy kernels
        processed_encoder = torch.empty((B, T, H), dtype=C.dtype, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=C.dtype, device=device)

        # Per-batch copies: encoder rows [0, B*T)
        for b in range(B):
            grid_e = (B * T, triton.cdiv(H, 64))
            copy_rows_to_encoder_kernel[grid_e](
                C, processed_encoder[b],
                B, T, I, H,
                C.stride(0), C.stride(1),
                processed_encoder[b].stride(0), processed_encoder[b].stride(1), processed_encoder[b].stride(2),
                b,
            )
        # Per-batch copies: hidden rows [B*T, M)
        for b in range(B):
            start_row = B * T + b * I
            end_row = start_row + I
            rows = list(range(start_row, end_row))
            grid_h = (len(rows), triton.cdiv(H, 64))
            copy_rows_to_hidden_kernel[grid_h](
                C, processed_hidden[b],
                B, T, I, H,
                C.stride(0), C.stride(1),
                processed_hidden[b].stride(0), processed_hidden[b].stride(1), processed_hidden[b].stride(2),
                b,
            )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
