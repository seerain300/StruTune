import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_sequences_kernel(
    enc_ptr,  # *float32, shape [B, T, H]
    hid_ptr,  # *float32, shape [B, I, H]
    out_ptr,  # *float32, shape [B, T+I, H]
    B, T, I, H,
    stride_eb, stride_et, stride_eh,  # enc strides
    stride_hb, stride_hi, stride_hh,  # hid strides
    stride_ob, stride_ot, stride_oh,  # out strides
):
    b = tl.program_id(0)
    # compute total sequence length
    L = T + I
    # loop over concatenated sequence positions
    for l in range(0, L):
        # masks for bounds (always true, but keeps code simple)
        mask = l < L
        # decide source
        if l < T:
            src_ptr = enc_ptr + b * stride_eb + l * stride_et
        else:
            src_ptr = hid_ptr + b * stride_hb + (l - T) * stride_hi
        # dst
        dst_ptr = out_ptr + b * stride_ob + l * stride_ot
        # load vector of size H
        # We assume H is small/moderate; do elementwise loop to avoid complicated vectorization
        for h in range(0, H):
            val = tl.load(src_ptr + h * stride_eh, mask=mask, other=0.0)
            tl.store(dst_ptr + h * stride_oh, val, mask=mask)
    # Note: We use masks as 1D; Triton loops over H are fine for typical H (e.g., 64, 128, 256, 512).
    # For very large H, consider vectorized loads/stores with tl.arange; here simplicity ensures correctness.


@triton.jit
def _matmul_gemm_triton(
    A_ptr,  # *float32, shape [M, K], M=B*(T+I), K=H, we pass A = out_cat.reshape(M, H)
    W_ptr,  # *float32, shape [H, H]
    C_ptr,  # *float32, shape [M, K]
    M, K,
    stride_am, stride_ak,  # A strides: (row, col) in flattened [M, K]
    stride_wn, stride_wk,  # W strides: (n, k) where n is hidden dim, k is hidden dim
    stride_cm, stride_ck,  # C strides: (row, col) in flattened [M, K]
    BLOCK_M: tl.constexpr,  # tile size in M
    BLOCK_N: tl.constexpr,  # tile size in N (here N==K)
    BLOCK_K: tl.constexpr,  # reduction tile in K
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # compute tile indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # load A tile: A[offs_m, offs_k]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        # load W^T tile: W^T has shape [K, K] with entries W[cols=n, k]. We load W[n, k].
        # We need C[offs_n, offs_k] = sum_{kk} A[offs_m, kk] * W[offs_n, kk]
        w_ptrs = W_ptr + (offs_n[None, :] * stride_wk + offs_k[:, None] * stride_wn)  # shape [BLOCK_K, BLOCK_N]
        w = tl.load(w_ptrs, mask=(offs_n[None, :] < K) & (offs_k[:, None] < K), other=0.0)
        # accumulate
        acc += tl.dot(a, w)  # a: [BLOCK_M, BLOCK_K], w: [BLOCK_K, BLOCK_N]
    # store acc to C
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_ck)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < K))


@triton.jit
def _split_outputs_kernel(
    C_ptr,  # *float32, shape [M, K], M=B*(T+I), K=H
    out_e_ptr,  # *float32, shape [B, T, H]
    out_i_ptr,  # *float32, shape [B, I, H]
    B, T, I, K,
    stride_cm, stride_ck,  # C strides
    stride_eb, stride_et, stride_eh,  # encoder output strides
    stride_ib, stride_ii, stride_ih,  # image output strides
):
    b = tl.program_id(0)
    # First, write encoder part: rows [0 .. B*T)
    for t in range(0, T):
        row = b * (T + I) + t
        # copy entire hidden vector
        for h in range(0, K):
            val = tl.load(C_ptr + row * stride_cm + h * stride_ck)
            tl.store(out_e_ptr + b * stride_eb + t * stride_et + h * stride_eh, val)
    # Then, write image part: rows [B*T .. B*(T+I))
    for i in range(0, I):
        row = b * (T + I) + T + i
        for h in range(0, K):
            val = tl.load(C_ptr + row * stride_cm + h * stride_ck)
            tl.store(out_i_ptr + b * stride_ib + i * stride_ii + h * stride_ih, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, H]
        encoder_hidden_states: [B, T, H]
        process_weight: [H, H] (no bias)
        Returns: (processed_encoder: [B, T, H], processed_hidden: [B, I, H])
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton"
        B, I, H = hidden_states.shape
        T, _, H2 = encoder_hidden_states.shape
        assert H2 == H, "hidden_dim mismatch"
        W = process_weight  # [H, H]
        device = hidden_states.device

        # 1) Concatenate sequences along sequence dimension into out_cat [B, T+I, H]
        out_cat = torch.empty((B, T + I, H), dtype=torch.float32, device=device)
        _concatenate_sequences_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            *encoder_hidden_states.stride(), *hidden_states.stride(), *out_cat.stride(),
            num_warps=1, num_stages=1,
        )

        # 2) Compute A = out_cat reshaped to [M, K], M = B*(T+I), K = H
        M = B * (T + I)
        K = H
        A = out_cat.reshape(M, K)

        # 3) Allocate output C [M, K] for GEMM
        C = torch.empty((M, K), dtype=torch.float32, device=device)

        # 4) Launch Triton GEMM: C = A @ W^T
        # We need to interpret W as W^T: use W[n, k] with strides (n=k dimension, k=n dimension).
        # For W [H, H], stride_wn = W.stride(1) (k), stride_wk = W.stride(0) (n).
        stride_wn = W.stride(1)  # typically 1
        stride_wk = W.stride(0)  # typically 1
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_cm = C.stride(0)
        stride_ck = C.stride(1)
        # Tile sizes: choose reasonable defaults; you can tune these for performance.
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32
        grid_m = (M + BLOCK_M - 1) // BLOCK_M
        grid_n = (K + BLOCK_N - 1) // BLOCK_N
        _matmul_gemm_triton[(grid_m, grid_n)](
            A, W, C,
            M, K,
            stride_am, stride_ak,
            stride_wk, stride_wn,  # note: pass (k, n) for W
            stride_cm, stride_ck,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 5) Split C back into [B, T, H] and [B, I, H] using Triton
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=device)
        _split_outputs_kernel[(B,)](
            C,
            processed_encoder, processed_hidden,
            B, T, I, K,
            *C.stride(),
            *processed_encoder.stride(), *processed_hidden.stride(),
            num_warps=1, num_stages=1,
        )

        return processed_encoder, processed_hidden