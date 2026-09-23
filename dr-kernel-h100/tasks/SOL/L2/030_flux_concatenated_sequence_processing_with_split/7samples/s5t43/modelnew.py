import torch
import triton
import triton.language as tl


@triton.jit
def copy_rows_kernel(
    src_ptr, dst_ptr,
    B: tl.int32, ROWS: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,
    dst_s0, dst_s1, dst_s2,
    ROW_START: tl.int32,  # starting row in src to copy into dst
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # Each program copies a tile of size (BLOCK_l rows) x (BLOCK_h cols) from src to dst
    # Mapping to batch and row indices:
    # grid dims: (B, ceil(ROWS/BLOCK_l), ceil(H/BLOCK_h))
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Row indices to copy (absolute in src)
    l = ROW_START + pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # [BLOCK_l]
    # Column indices (hidden dim)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)             # [BLOCK_h]

    # Masks for valid rows and cols
    mask_l = l < (ROW_START + ROWS)
    mask_h = h < H
    mask = mask_l[:, None] & mask_h[None, :]

    # Compute base pointers for the batch
    base_src = src_ptr + pid_b * src_s0
    base_dst = dst_ptr + pid_b * dst_s0

    # Compute element pointers for src and dst
    # src index for each (l, h): src_s0*pid_b + src_s1*l + src_s2*h
    # dst index for each (l-ROW_START, h): dst_s0*pid_b + dst_s1*(l-ROW_START) + dst_s2*h
    src_ptrs = base_src + l[:, None] * src_s1 + h[None, :] * src_s2
    # dst row is (l - ROW_START)
    dst_row = l - ROW_START
    dst_ptrs = base_dst + dst_row[:, None] * dst_s1 + h[None, :] * dst_s2

    # Load and store with mask
    vals = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.store(dst_ptrs, vals, mask=mask)


@triton.jit
def matmul_kernel(
    A_ptr, W_ptr, C_ptr,
    M: tl.int32, K: tl.int32, N: tl.int32,
    A_s0, A_s1, A_s2,
    W_s0, W_s1, W_s2,  # W is [K, N]
    C_s0, C_s1, C_s2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling across output matrix (M rows, N cols)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Masks for valid output rows/cols
    mask_m = m < M
    mask_n = n < N
    mask_out = mask_m[:, None] & mask_n[None, :]

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k < K

        # Load A tile: A is [M, K], pointer arithmetic:
        # A_ptr + m[:, None]*A_s1 + k[None, :]*A_s2
        A_ptrs = A_ptr + m[:, None] * A_s1 + k[None, :] * A_s2
        A_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load W^T tile: W is [K, N], but we access as W^T [N, K] by swapping indices:
        # W^T_ptrs[i, j] = W_ptr + k[j]*W_s0 + n[i]*W_s1
        WT_ptrs = W_ptr + k[:, None] * W_s0 + n[None, :] * W_s1
        WT_mask = mask_k[:, None] & mask_n[None, :]
        w_t = tl.load(WT_ptrs, mask=WT_mask, other=0.0)

        # Accumulate in float32
        acc += tl.dot(a.to(tl.float32), w_t.to(tl.float32))

    # Store result to C: C is [M, N], pointer arithmetic:
    # C_ptr + m[:, None]*C_s1 + n[None, :]*C_s2
    C_ptrs = C_ptr + m[:, None] * C_s1 + n[None, :] * C_s2
    tl.store(C_ptrs, acc, mask=mask_out)


def _triton_concat_copy(encoder: torch.Tensor, image: torch.Tensor, out: torch.Tensor):
    """
    Write out = [encoder, image] along sequence dimension using Triton kernels.
    encoder: [B, T, H], image: [B, I, H], out: [B, T+I, H]
    """
    B, T, H = encoder.shape
    B2, I, H2 = image.shape
    assert B == B2 and H == H2, "encoder and image must have matching batch and hidden dims"
    # Kernel 1: copy encoder rows into out[:, :T, :]
    grid_enc = (B, triton.cdiv(T, 64), triton.cdiv(H, 64))
    copy_rows_kernel[grid_enc](
        encoder, out,
        B, T, H,
        encoder.stride(0), encoder.stride(1), encoder.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        ROW_START=0, BLOCK_l=64, BLOCK_h=64,
        num_warps=4, num_stages=2,
    )
    # Kernel 2: copy image rows into out[:, T:, :]
    grid_img = (B, triton.cdiv(I, 64), triton.cdiv(H, 64))
    copy_rows_kernel[grid_img](
        image, out,
        B, I, H,
        image.stride(0), image.stride(1), image.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        ROW_START=T, BLOCK_l=64, BLOCK_h=64,
        num_warps=4, num_stages=2,
    )


def _triton_split_copy(src: torch.Tensor, out: torch.Tensor, row_start: int, count: int):
    """
    Copy rows [row_start: row_start+count) from src into out using Triton.
    src: [B, L, H], out: [B, count, H]
    """
    B, L, H = src.shape
    grid = (B, triton.cdiv(count, 64), triton.cdiv(H, 64))
    copy_rows_kernel[grid](
        src, out,
        B, count, H,
        src.stride(0), src.stride(1), src.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        ROW_START=row_start, BLOCK_l=64, BLOCK_h=64,
        num_warps=4, num_stages=2,
    )


def _triton_matmul(A: torch.Tensor, W: torch.Tensor, C: torch.Tensor):
    """
    Compute C = A @ W.T using Triton. A: [M, K], W: [K, N], C: [M, N]
    """
    M, K = A.shape
    K_w, N = W.shape
    assert K == K_w, "A's second dim must match W's first dim"
    # Choose tiling parameters; simple defaults that work broadly
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        A, W, C,
        M, K, N,
        A.stride(0), A.stride(1), A.stride(2),
        W.stride(0), W.stride(1), W.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton implementation:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dim into 'concatenated'.
        2) Compute processed = concatenated @ process_weight.T using Triton.
        3) Split processed into processed_encoder and processed_hidden.
        Returns (processed_encoder, processed_hidden).
        """
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"

        # Shapes
        B, T, H = encoder_hidden_states.shape
        B2, I, H2 = hidden_states.shape
        assert B == B2 and H == H2, "Batch and hidden dims must match for concatenation"
        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1) Concatenate using Triton copy kernels
        concatenated = torch.empty((B, T + I, H), device=device, dtype=dtype)
        _triton_concat_copy(encoder_hidden_states, hidden_states, concatenated)

        # 2) Matmul processed = concatenated @ process_weight.T
        # concatenated: [B, L, H], process_weight: [H, H]
        # Let M = B * L, K = H, N = H
        L = T + I
        A = concatenated  # [B, L, H]
        W = process_weight  # [H, H]
        # We need C as [B, L, H] = [M, K] @ [K, N], but here N == K == H.
        # To feed matmul kernel, we treat:
        # A as [M=L*B, K=H], W as [K=H, N=H]
        # Allocate output C
        processed = torch.empty((B, L, H), device=device, dtype=torch.float32)  # compute in fp32
        _triton_matmul(A, W, processed)

        # Cast back to original dtype if needed
        processed = processed.to(dtype)

        # 3) Split processed into encoder and hidden parts
        processed_encoder = torch.empty((B, T, H), device=device, dtype=dtype)
        _triton_split_copy(processed, processed_encoder, row_start=0, count=T)

        processed_hidden = torch.empty((B, I, H), device=device, dtype=dtype)
        _triton_split_copy(processed, processed_hidden, row_start=T, count=I)

        return processed_encoder, processed_hidden