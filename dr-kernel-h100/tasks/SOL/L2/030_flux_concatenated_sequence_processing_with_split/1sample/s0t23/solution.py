import torch
import triton
import triton.language as tl

# 1) Triton kernel: concatenate encoder_hidden_states and hidden_states along sequence dimension.
# Inputs:
#   enc: [B, T, H], strided
#   hid: [B, I, H], strided
#   out_cat: [B, T+I, H], strided
@triton.jit
def _concatenate_seqs_kernel(
    enc_ptr, hid_ptr, out_ptr,
    B, T, I, H,
    enc_stride_b, enc_stride_t, enc_stride_h,
    hid_stride_b, hid_stride_i, hid_stride_h,
    out_stride_b, out_stride_l, out_stride_h,
    BLOCK_L: tl.constexpr,
):
    b = tl.program_id(0)
    # out_cat[b, l, h] = enc[b, l, h] if l < T else hid[b, l - T, h]
    # Loop over sequence length
    for l in range(0, T + I):
        h_offsets = tl.arange(0, H)
        mask_h = h_offsets < H
        # Determine source tensor
        src_ptr = enc_ptr + b * enc_stride_b + l * enc_stride_t + h_offsets * enc_stride_h if l < T \
                  else hid_ptr + b * hid_stride_b + (l - T) * hid_stride_i + h_offsets * hid_stride_h
        vals = tl.load(src_ptr, mask=mask_h, other=0.0)
        dst_ptr = out_ptr + b * out_stride_b + l * out_stride_l + h_offsets * out_stride_h
        tl.store(dst_ptr, vals, mask=mask_h)


# 2) Triton kernel: batched GEMM computing C = A[M, H] @ W[H, H]^T, where A is the concatenated tensor [B*(T+I), H].
# We use a 2D grid over output rows (M) and columns (H). To ensure correctness, we set BLOCK_N = H as meta parameter,
# so each program computes a full row-block across all H columns. This avoids missing columns in the grid.
@triton.jit
def _batched_gemm_right_kernel_2d(
    A_ptr, W_ptr, C_ptr,
    M, H,
    A_stride_m, A_stride_k,
    W_stride_k, W_stride_n,  # W is [K=H, N=H]
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,    # set BLOCK_N = H at launch for correctness
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Rows and columns this program computes
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # equals [0..H-1] when BLOCK_N=H

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K dimension
    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        # Load A tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + rows[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        a_mask = (rows[:, None] < M) & (k_offsets[None, :] < H)
        A_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load W^T tile as W[k, n]: shape [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + k_offsets[:, None] * W_stride_k + cols[None, :] * W_stride_n
        w_mask = (k_offsets[:, None] < H) & (cols[None, :] < H)
        W_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, W_tile)

    # Store results: C[row, col] = acc[row, col]
    c_ptrs = C_ptr + rows[:, None] * C_stride_m + cols[None, :] * C_stride_n
    store_mask = (rows[:, None] < M) & (cols[None, :] < H)
    tl.store(c_ptrs, acc, mask=store_mask)


# 3) Triton kernel: split C[M, H] into processed_encoder [B, T, H] and processed_hidden [B, I, H].
# One program per batch. We iterate over rows and columns to copy the appropriate segments.
@triton.jit
def _split_streams_kernel(
    C_ptr, enc_ptr, hid_ptr,
    M, T, I, H,
    C_stride_m, C_stride_n,
    enc_stride_b, enc_stride_t, enc_stride_h,
    hid_stride_b, hid_stride_i, hid_stride_h,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    # Copy rows 0..T-1 to encoder output
    for t in range(0, T):
        row = b * (T + I) + t
        if row >= M:
            break
        for n in range(0, H, BLOCK):
            col = n + tl.arange(0, BLOCK)
            c_ptrs = C_ptr + row * C_stride_m + col * C_stride_n
            mask = col < H
            vals = tl.load(c_ptrs, mask=mask, other=0.0)
            dst_ptr = enc_ptr + b * enc_stride_b + t * enc_stride_t + col * enc_stride_h
            tl.store(dst_ptr, vals, mask=mask)

    # Copy rows T..T+I-1 to hidden output
    for i in range(0, I):
        row = b * (T + I) + T + i
        if row >= M:
            break
        for n in range(0, H, BLOCK):
            col = n + tl.arange(0, BLOCK)
            c_ptrs = C_ptr + row * C_stride_m + col * C_stride_n
            mask = col < H
            vals = tl.load(c_ptrs, mask=mask, other=0.0)
            dst_ptr = hid_ptr + b * hid_stride_b + i * hid_stride_i + col * hid_stride_h
            tl.store(dst_ptr, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation of:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dimension
        2) Apply linear projection: concatenated @ process_weight.T
        3) Split back into separate encoder and image streams
        Returns (processed_encoder_hidden_states, processed_hidden_states)
        """
        # Ensure inputs are on the same device and dtype
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors"
        B, I, H = hidden_states.shape
        assert encoder_hidden_states.shape[0] == B and encoder_hidden_states.shape[2] == H, "encoder_hidden_states must have shape [B, T, H]"
        T = encoder_hidden_states.shape[1]
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"
        dtype = process_weight.dtype  # use same dtype as weight; typically float32

        device = hidden_states.device

        # 1) Concatenate using Triton
        out_cat = torch.empty((B, T + I, H), dtype=dtype, device=device)
        _concatenate_seqs_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            *encoder_hidden_states.stride(), *hidden_states.stride(),
            *out_cat.stride(),
            BLOCK_L=max(T + I, 1),
            num_warps=1, num_stages=1,
        )

        # 2) GEMM in Triton: C = out_cat @ process_weight.T
        # out_cat: [M, H], M = B*(T+I)
        M = B * (T + I)
        A = out_cat  # [M, H], contiguous in last dim
        W = process_weight  # [H, H]
        C = torch.empty((M, H), dtype=torch.float32, device=device)  # accumulate in float32

        # Use 2D grid over rows and columns; set BLOCK_N = H to ensure full columns are computed
        BLOCK_M = 128
        BLOCK_N = H  # meta-parameter must be known at compile time per launch
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        _batched_gemm_right_kernel_2d[grid](
            A, W, C,
            M, H,
            *A.stride(), *W.stride(), *C.stride(),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split into two outputs using Triton
        processed_encoder = torch.empty((B, T, H), dtype=dtype, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=dtype, device=device)
        _split_streams_kernel[(B,)](
            C, processed_encoder, processed_hidden,
            M, T, I, H,
            *C.stride(),
            *processed_encoder.stride(), *processed_hidden.stride(),
            BLOCK=128,
            num_warps=1, num_stages=1,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
