import torch
import triton
import triton.language as tl


# Kernel: build concatenated input X_cat[b] row-by-row.
# Input:
#   encoder_hidden_states: [B, T, H]
#   hidden_states:          [B, I, H]
# Output:
#   X_cat[b]: [T + I, H] where rows 0..T-1 come from encoder_hidden_states[b], and rows T..T+I-1 come from hidden_states[b]
@triton.jit
def kernel_cat_rowwise(
    e_ptr, i_ptr, out_ptr,
    B, T, I, H,
    stride_eb, stride_et, stride_eh,
    stride_ib, stride_it, stride_ih,
    stride_ob, stride_om, stride_on,
):
    pid_b = tl.program_id(0)
    pid_p = tl.program_id(1)  # sequence position in concatenated output [0, T+I)
    if pid_b >= B or pid_p >= (T + I):
        return

    if pid_p < T:
        src = 0  # encoder
        row_idx = pid_p
    else:
        src = 1  # hidden
        row_idx = pid_p - T

    # Compute base pointers
    if src == 0:
        base = e_ptr + pid_b * stride_eb
        row_base = base + row_idx * stride_et
    else:
        base = i_ptr + pid_b * stride_ib
        row_base = base + row_idx * stride_it

    # Write the row into out[b, pid_p, :]
    out_base = out_ptr + pid_b * stride_ob + pid_p * stride_om
    for j in range(0, H):
        val = tl.load(row_base + j * stride_eh)
        tl.store(out_base + j * stride_on, val)


# Kernel: batched matmul for each batch b: Y[b] = X_cat[b] @ W
# X_cat[b]: [M, K] where M = T + I, K = H
# W:        [K, N] where N = H (here K == N == H, so we can use W directly)
# Y[b]:     [M, N]
@triton.jit
def kernel_batched_matmul(
    X_ptr, W_ptr, Y_ptr,
    B, M, K, N,
    stride_xb, stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_yb, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    if pid_b >= B:
        return

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A block: X[pid_m, k0:k0+BLOCK_K] -> shape (BLOCK_M, BLOCK_K)
        a_ptrs = X_ptr + pid_b * stride_xb + m_start * stride_xm + k_offsets[None, :] * stride_xk
        a_mask = (m_start + tl.arange(0, BLOCK_M))[:, None] < M
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B block: W[k0:k0+BLOCK_K, n_start:n_start+BLOCK_N] -> shape (BLOCK_K, BLOCK_N)
        b_ptrs = W_ptr + k_offsets[:, None] * stride_wk + (n_start + tl.arange(0, BLOCK_N))[None, :] * stride_wn
        b_mask = (n_start + tl.arange(0, BLOCK_N))[None, :] < N
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store results
    y_ptrs = Y_ptr + pid_b * stride_yb + (m_start + tl.arange(0, BLOCK_M))[:, None] * stride_ym + (n_start + tl.arange(0, BLOCK_N))[None, :] * stride_yn
    y_mask = (m_start + tl.arange(0, BLOCK_M))[:, None] < M
    tl.store(y_ptrs, acc, mask=y_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure CUDA tensors for Triton
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors for Triton kernels."

        B = hidden_states.shape[0]
        assert B == encoder_hidden_states.shape[0], "Batch sizes must match."
        H = hidden_states.shape[2]
        assert H == encoder_hidden_states.shape[2], "Hidden dimensions must match."
        assert H == process_weight.shape[0] and H == process_weight.shape[1], "process_weight must be square [H, H]."

        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        M = T + I

        # Allocate concatenated input per batch
        X_cat = [torch.empty((M, H), device=hidden_states.device, dtype=hidden_states.dtype) for _ in range(B)]

        # Launch cat kernel: grid = (B, M)
        grid_cat = (B, M)
        stride_eb, stride_et, stride_eh = encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2)
        stride_ib, stride_it, stride_ih = hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2)
        stride_ob, stride_om, stride_on = X_cat[0].stride(0), X_cat[0].stride(1), X_cat[0].stride(2)
        # Note: we use out_ptr as a pointer to X_cat[b], Triton will handle indexing via pid_b and pid_p.
        # To pass X_cat properly, we need to pass each tensor individually; Triton allows calling with separate arguments.
        # So we call the kernel once per batch by iterating over b and launching with appropriate out_ptr.
        for b in range(B):
            out = X_cat[b]
            kernel_cat_rowwise[(1, M)](
                encoder_hidden_states[b], hidden_states[b], out,
                1, T, I, H,
                stride_eb, stride_et, stride_eh,
                stride_ib, stride_it, stride_ih,
                stride_ob, stride_om, stride_on,
                num_warps=1, num_stages=1,
            )

        # Prepare output Y per batch
        Y = [torch.empty((M, H), device=hidden_states.device, dtype=hidden_states.dtype) for _ in range(B)]

        # Launch batched matmul kernel: grid = (B, tiles over M, tiles over N)
        # Choose block sizes based on runtime shapes (simple heuristic)
        BLOCK_M = 128 if M >= 128 else 64
        BLOCK_N = 64 if H >= 64 else 32
        BLOCK_K = 32 if H >= 32 else 16

        grid_mm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        for b in range(B):
            X_b = X_cat[b]
            Y_b = Y[b]
            W = process_weight  # [H, H], no bias

            # Strides for X_b
            stride_xb = X_b.stride(0)
            stride_xm = X_b.stride(1)
            stride_xk = X_b.stride(2)

            # Strides for W
            stride_wk = W.stride(0)
            stride_wn = W.stride(1)

            # Strides for Y_b
            stride_yb = Y_b.stride(0)
            stride_ym = Y_b.stride(1)
            stride_yn = Y_b.stride(2)

            kernel_batched_matmul[grid_mm](
                X_b, W, Y_b,
                B, M, H, H,
                stride_xb, stride_xm, stride_xk,
                stride_wk, stride_wn,
                stride_yb, stride_ym, stride_yn,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=2,
            )

        # Split outputs: processed_encoder [B, T, H], processed_hidden [B, I, H]
        processed_encoder_list = [Y_b[:T, :] for b, Y_b in enumerate(Y)]
        processed_hidden_list = [Y_b[T:, :] for b, Y_b in enumerate(Y)]
        processed_encoder = torch.stack(processed_encoder_list, dim=0) if len(processed_encoder_list) > 0 else torch.empty((0, T, H), device=hidden_states.device, dtype=hidden_states.dtype)
        processed_hidden = torch.stack(processed_hidden_list, dim=0) if len(processed_hidden_list) > 0 else torch.empty((0, I, H), device=hidden_states.device, dtype=hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
