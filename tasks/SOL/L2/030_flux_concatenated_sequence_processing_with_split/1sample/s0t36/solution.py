import torch
import triton
import triton.language as tl

# Triton kernel: concatenate [B, T, H] and [B, I, H] into [B, T+I, H]
@triton.jit
def _concatenate_sequences_kernel(
    enc_ptr,           # *ptr to encoder_hidden_states [B, T, H]
    hid_ptr,           # *ptr to hidden_states [B, I, H]
    out_ptr,           # *ptr to output [B, T+I, H]
    B, T, I, H,
    enc_stride_b, enc_stride_t, enc_stride_h,
    hid_stride_b, hid_stride_i, hid_stride_h,
    out_stride_b, out_stride_t, out_stride_h,
):
    b = tl.program_id(0)
    # bounds check for batch
    if b >= B:
        return
    # loop over concatenated sequence length
    total = T + I
    # Triton loop: iterate l from 0 to total-1
    for l in range(0, total):
        # determine source: first T elements from encoder, remaining from hidden
        is_encoder = l < T
        # compute source pointers
        # encoder: enc[b, l, :]
        enc_row_ptr = enc_ptr + b * enc_stride_b + l * enc_stride_t
        # hidden: hid[b, l - T, :]
        hid_row_ptr = hid_ptr + b * hid_stride_b + (l - T) * hid_stride_i
        # output pointer
        out_row_ptr = out_ptr + b * out_stride_b + l * out_stride_t
        # load and store: both loads masked by is_encoder (scalar boolean), but Triton expects tensors
        # We'll construct per-column vectors for h in [0, H)
        h = tl.arange(0, H)  # vector of column indices
        # load
        if is_encoder:
            vals = tl.load(enc_row_ptr + h * enc_stride_h)
        else:
            vals = tl.load(hid_row_ptr + h * hid_stride_h)
        # store
        tl.store(out_row_ptr + h * out_stride_h, vals)

# Triton kernel: batched matmul
# Computes C[M, N] = A[M, K] @ W[K, N], here A is out_cat flattened to [M, K] with M=B*(T+I), K=H, N=H
# We need right-multiply by process_weight.T, so W is [H, H] and we multiply A @ W (since W.T is [K, N]).
@triton.jit
def _batched_gemm_right_kernel(
    A_ptr,  # *ptr to A [M, K]
    W_ptr,  # *ptr to W [H, H] (right-multiply)
    C_ptr,  # *ptr to C [M, H]
    M, K, N,
    A_stride_m, A_stride_k,
    W_stride_k, W_stride_n,
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr,  # tile size along M (rows of output)
    BLOCK_N: tl.constexpr,  # tile size along N (columns of output)
    BLOCK_K: tl.constexpr,  # reduction tile size along K
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # compute tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows in output
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols in output
    offs_k = tl.arange(0, BLOCK_K)  # reduction dim

    # Create pointers for the first K tile
    # A is [M, K], W is [K, N] (note: W is passed as [H, H], we treat it as [K, N] where K=N=H)
    A_ptrs = A_ptr + (offs_m[:, None] * A_stride_m) + (offs_k[None, :] * A_stride_k)
    W_ptrs = W_ptr + (offs_k[:, None] * W_stride_k) + (offs_n[None, :] * W_stride_n)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    # K can be <= N (here K=N=H), but we keep the generic loop
    k0 = 0
    while k0 < K:
        # Masks for bounds (robustness)
        a_mask = (offs_m[:, None] < M) & (k0 + offs_k[None, :] < K)
        w_mask = (k0 + offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)
        w = tl.load(W_ptrs, mask=w_mask, other=0.0)
        # a: [BLOCK_M, BLOCK_K], w: [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, w)
        # advance K tile
        k0 += BLOCK_K
        # update pointers
        A_ptrs += BLOCK_K * A_stride_k
        W_ptrs += BLOCK_K * W_stride_k

    # Write back to C
    c_ptrs = C_ptr + (offs_m[:, None] * C_stride_m) + (offs_n[None, :] * C_stride_n)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

# Triton kernel: split C[M, H] into [B, T, H] and [B, I, H]
@triton.jit
def _split_streams_kernel(
    C_ptr,             # *ptr to C [M, H]
    out_encoder_ptr,   # *ptr to processed_encoder [B, T, H]
    out_hidden_ptr,    # *ptr to processed_hidden [B, I, H]
    M, T, I, H,
    C_stride_m, C_stride_n,
    enc_stride_b, enc_stride_t, enc_stride_h,
    hid_stride_b, hid_stride_i, hid_stride_h,
):
    b = tl.program_id(0)
    if b >= B:
        return
    # For each sequence position: write to encoder or hidden output accordingly
    total = T + I
    # We'll loop over columns (hidden_dim) vectorized; since H is typically small, this is fine.
    for l in range(0, total):
        is_encoder = l < T
        # compute output pointers
        # output rows are flattened: out_encoder[b, l, :] and out_hidden[b, l - T, :]
        out_encoder_row_ptr = out_encoder_ptr + b * enc_stride_b + l * enc_stride_t
        out_hidden_row_ptr = out_hidden_ptr + b * hid_stride_b + (l - T) * hid_stride_i
        # C row index: m = b * (T + I) + l
        m = b * (T + I) + l
        # load from C[m, :]
        h = tl.arange(0, H)
        c_ptrs = C_ptr + m * C_stride_m + h * C_stride_n
        vals = tl.load(c_ptrs)
        if is_encoder:
            tl.store(out_encoder_row_ptr + h * enc_stride_h, vals)
        else:
            tl.store(out_hidden_row_ptr + h * hid_stride_h, vals)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, H]
        encoder_hidden_states: [B, T, H]
        process_weight: [H, H]
        Returns (processed_encoder: [B, T, H], processed_hidden: [B, I, H])
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "inputs must be 3D tensors"
        B = hidden_states.shape[0]
        I = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "hidden_dim must match"
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"
        device = hidden_states.device

        # 1) Concatenate along sequence dimension using Triton
        out_cat = torch.empty((B, T + I, H), dtype=hidden_states.dtype, device=device)
        # Strides
        enc_stride_b, enc_stride_t, enc_stride_h = encoder_hidden_states.stride()
        hid_stride_b, hid_stride_i, hid_stride_h = hidden_states.stride()
        out_stride_b, out_stride_t, out_stride_h = out_cat.stride()
        _concatenate_sequences_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            enc_stride_b, enc_stride_t, enc_stride_h,
            hid_stride_b, hid_stride_i, hid_stride_h,
            out_stride_b, out_stride_t, out_stride_h,
            num_warps=1, num_stages=1,
        )

        # 2) Batched GEMM using Triton: C[M, H] = out_cat[M, H] @ process_weight[H, H]
        # Treat out_cat as [M, K] where M = B*(T+I), K = H, and W as [K, N] = [H, H]
        M = B * (T + I)
        K = H
        N = H  # right-multiply by HxH
        C = torch.empty((M, N), dtype=torch.float32, device=device)  # accumulate in fp32
        # Strides for A (out_cat), W (process_weight), and C
        # out_cat is contiguous [M, H] after concat; we can flatten pointer
        # W is [H, H], contiguous; we'll pass as [K, N] with K=N=H
        # For A, use flattened layout: A_stride_m=H, A_stride_k=1 (contiguous columns)
        A_stride_m = H
        A_stride_k = 1
        W_stride_k = 1  # process_weight is contiguous along rows
        W_stride_n = H  # contiguous along columns
        C_stride_m = 1
        C_stride_n = H

        # Choose tile sizes; these are good defaults for many shapes
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        grid_m = triton.cdiv(M, BLOCK_M)
        grid_n = triton.cdiv(N, BLOCK_N)

        _batched_gemm_right_kernel[(grid_m, grid_n)](
            out_cat, process_weight, C,
            M, K, N,
            A_stride_m, A_stride_k,
            W_stride_k, W_stride_n,
            C_stride_m, C_stride_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split streams into [B, T, H] and [B, I, H] using Triton
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=device)
        # Strides for outputs (they are contiguous, but pass strides for generality)
        enc_stride_b_out, enc_stride_t_out, enc_stride_h_out = processed_encoder.stride()
        hid_stride_b_out, hid_stride_i_out, hid_stride_h_out = processed_hidden.stride()
        _split_streams_kernel[(B,)](
            C,
            processed_encoder, processed_hidden,
            B, T, I, H,
            C.stride(0), C.stride(1),
            enc_stride_b_out, enc_stride_t_out, enc_stride_h_out,
            hid_stride_b_out, hid_stride_i_out, hid_stride_h_out,
            num_warps=1, num_stages=1,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
