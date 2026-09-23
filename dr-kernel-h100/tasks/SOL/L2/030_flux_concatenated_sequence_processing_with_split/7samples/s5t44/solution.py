import torch
import triton
import triton.language as tl


@triton.jit
def _copy_rows_kernel(
    src_ptr, dst_ptr,
    B: tl.int32, COUNT: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,
    dst_s0, dst_s1, dst_s2,
    ROW_START: tl.int32,
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # grid: (B, tiles over rows, tiles over H)
    pid_b = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_h = tl.program_id(2)

    l = ROW_START + pid_r * BLOCK_l + tl.arange(0, BLOCK_l)  # sequence row indices to copy
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)              # hidden dim indices

    # mask: valid batch and valid l and valid h
    mask = (l < (ROW_START + COUNT)) & (h < H)

    # base pointers for this batch
    src_batch_ptr = src_ptr + pid_b * src_s0

    # Compute pointers for each (l, h) and copy
    # Using broadcasting: [BLOCK_l, 1] * [1, BLOCK_h]
    src_ptrs = src_batch_ptr + l[:, None] * src_s1 + h[None, :] * src_s2
    dst_ptrs = dst_ptr + pid_b * dst_s0 + l[:, None] * dst_s1 + h[None, :] * dst_s2

    # Load and store with mask
    vals = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.store(dst_ptrs, vals, mask=mask)


@triton.jit
def _matmul_kernel(
    A_ptr, W_ptr, C_ptr,
    B: tl.int32, T_I: tl.int32, H: tl.int32,
    A_s0, A_s1, A_s2,
    W_s0, W_s1,
    C_s0, C_s1, C_s2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Compute C = A @ W^T, where A is [B, T_I, H], W is [H, H], C is [B, T_I, H]
    # We flatten M = B * T_I
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # row index in flattened M
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # column index in N=H

    # Mask for valid rows/cols
    mask_m = m < (B * T_I)
    mask_n = n < H

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K=H in chunks
    for k in range(0, H, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)  # reduction indices
        mask_k = kk < H

        # Map flattened m to (b, l)
        b = m // T_I
        l = m % T_I

        # A pointers: shape (BLOCK_M, BLOCK_K)
        A_ptrs = A_ptr + b[:, None] * A_s0 + l[:, None] * A_s1 + kk[None, :] * A_s2
        # W^T pointers: W is [H, H]; W^T[kk, kk] is [BLOCK_K, BLOCK_N] (kk reduction, n output)
        Wt_ptrs = W_ptr + kk[:, None] * W_s0 + n[None, :] * W_s1

        # Load with masks
        a = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        wt = tl.load(Wt_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(a, wt)

    # Write back to C
    # Map m back to (b, l)
    b_out = m // T_I
    l_out = m % T_I

    C_ptrs = C_ptr + b_out[:, None] * C_s0 + l_out[:, None] * C_s1 + n[None, :] * C_s2
    store_mask = mask_m[:, None] & mask_n[None, :]
    # Store as float32 (original model typically uses float32)
    tl.store(C_ptrs, acc, mask=store_mask)


def _launch_copy_concat(b, T: int, I: int, H: int, device, dtype):
    # Concatenated processed tensor
    processed = torch.empty((b, T + I, H), device=device, dtype=dtype)

    # Copy encoder_hidden_states -> processed[:, :T, :]
    grid_encoder = (b, triton.cdiv(T, 64), triton.cdiv(H, 64))
    _copy_rows_kernel[grid_encoder](
        src_ptr=encoder_hidden_states, dst_ptr=processed,
        B=b, COUNT=T, H=H,
        src_s0=encoder_hidden_states.stride(0), src_s1=encoder_hidden_states.stride(1), src_s2=encoder_hidden_states.stride(2),
        dst_s0=processed.stride(0), dst_s1=processed.stride(1), dst_s2=processed.stride(2),
        ROW_START=0,
        BLOCK_l=64, BLOCK_h=64,
        num_warps=4, num_stages=2,
    )

    # Copy hidden_states -> processed[:, T:, :]
    grid_image = (b, triton.cdiv(I, 64), triton.cdiv(H, 64))
    _copy_rows_kernel[grid_image](
        src_ptr=hidden_states, dst_ptr=processed,
        B=b, COUNT=I, H=H,
        src_s0=hidden_states.stride(0), src_s1=hidden_states.stride(1), src_s2=hidden_states.stride(2),
        dst_s0=processed.stride(0), dst_s1=processed.stride(1), dst_s2=processed.stride(2),
        ROW_START=T,
        BLOCK_l=64, BLOCK_h=64,
        num_warps=4, num_stages=2,
    )

    return processed


def _launch_matmul(processed, process_weight, device, dtype):
    B = processed.shape[0]
    T_I = processed.shape[1]
    H = processed.shape[2]
    W = process_weight  # [H, H]
    # Output C = processed @ W^T, C is [B, T_I, H]
    C = torch.empty((B, T_I, H), device=device, dtype=torch.float32)  # compute in fp32 for stability

    # Launch matmul kernel
    # Choose moderate tile sizes; can be tuned later
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(B * T_I, BLOCK_M), triton.cdiv(H, BLOCK_N))
    _matmul_kernel[grid](
        A_ptr=processed, W_ptr=W,
        C_ptr=C,
        B=B, T_I=T_I, H=H,
        A_s0=processed.stride(0), A_s1=processed.stride(1), A_s2=processed.stride(2),
        W_s0=W.stride(0), W_s1=W.stride(1),
        C_s0=C.stride(0), C_s1=C.stride(1), C_s2=C.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    # Cast back to original dtype if needed
    if dtype != torch.float32:
        C = C.to(dtype)
    return C


def _triton_split_copy(src, dst, row_start: int, count: int):
    b = src.shape[0]
    h = src.shape[2]
    # Launch copy kernel
    grid = (b, triton.cdiv(count, 64), triton.cdiv(h, 64))
    _copy_rows_kernel[grid](
        src_ptr=src, dst_ptr=dst,
        B=b, COUNT=count, H=h,
        src_s0=src.stride(0), src_s1=src.stride(1), src_s2=src.stride(2),
        dst_s0=dst.stride(0), dst_s1=dst.stride(1), dst_s2=dst.stride(2),
        ROW_START=row_start,
        BLOCK_l=64, BLOCK_h=64,
        num_warps=4, num_stages=2,
    )


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure CUDA tensors; if not on CUDA, move to current device
        device = hidden_states.device
        if encoder_hidden_states.device != device:
            encoder_hidden_states = encoder_hidden_states.to(device)
        if process_weight.device != device:
            process_weight = process_weight.to(device)
        dtype = hidden_states.dtype

        b = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]

        # 1) Concatenate along sequence dimension using Triton (two copy kernels)
        processed = _launch_copy_concat(b, T, I, H, device, dtype)

        # 2) Matmul: processed @ process_weight.T using Triton
        # Note: process_weight is [H, H]; ensure contiguous
        if not process_weight.is_contiguous():
            process_weight = process_weight.contiguous()
        processed = processed.contiguous()
        C = _launch_matmul(processed, process_weight, device, dtype)

        # 3) Split into two streams using Triton copy kernels
        processed_encoder = torch.empty((b, T, H), device=device, dtype=dtype)
        processed_hidden = torch.empty((b, I, H), device=device, dtype=dtype)

        _triton_split_copy(C, processed_encoder, row_start=0, count=T)
        _triton_split_copy(C, processed_hidden, row_start=T, count=I)

        return processed_encoder, processed_hidden

# The following helper functions mirror the original signatures for evaluation harnesses
@torch.no_grad()
def run(hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model = ModelNew()
    return model.forward(hidden_states, encoder_hidden_states, process_weight)


# Optional: keep get_inputs helper if needed by evaluator
def get_inputs():
    # Example random inputs; evaluator may override
    B = 2
    H = 64
    T = 128
    I = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_states = torch.randn(B, I, H, device=device, dtype=torch.float32)
    encoder_hidden_states = torch.randn(B, T, H, device=device, dtype=torch.float32)
    process_weight = torch.randn(H, H, device=device, dtype=torch.float32)
    return hidden_states, encoder_hidden_states, process_weight

def get_init_inputs():
    return []


def run(*args):
    return ModelNew()(*args)
