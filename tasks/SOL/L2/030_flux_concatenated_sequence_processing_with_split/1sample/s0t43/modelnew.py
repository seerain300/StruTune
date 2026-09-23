import torch
import triton
import triton.language as tl

# Kernel 1: Concatenate encoder_hidden_states and hidden_states along the sequence dimension.
@triton.jit
def _concatenation_kernel(
    encoder_ptr,  # *ptr to [B, T, H]
    hidden_ptr,   # *ptr to [B, I, H]
    out_ptr,      # *ptr to [B, T+I, H]
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    H: tl.constexpr,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_i, stride_h_h,
    stride_o_b, stride_o_l, stride_o_h,
):
    b = tl.program_id(0)
    # Compute total sequence length
    L = T + I
    # Loop over sequence positions l
    for l in range(0, L):
        # Determine source tensor
        if l < T:
            src_ptr = encoder_ptr + b * stride_e_b + l * stride_e_t
        else:
            src_ptr = hidden_ptr + b * stride_h_b + (l - T) * stride_h_i
        # Load row vector of length H
        # We need a vector of indices for hidden dimension
        h_idx = tl.arange(0, H)
        # Load with mask to avoid OOB (in case H isn't a multiple of block)
        mask = h_idx < H
        vals = tl.load(src_ptr + h_idx * stride_e_h, mask=mask, other=0.0)
        # Store to output
        dst_ptr = out_ptr + b * stride_o_b + l * stride_o_l
        tl.store(dst_ptr + h_idx * stride_o_h, vals, mask=mask)


# Kernel 2: Batched GEMM in Triton: C[M, K] = A[M, K] @ W[K, K]^T
@triton.jit
def _batched_matmul_kernel(
    A_ptr,    # *ptr to [M, K], where M=B*(T+I), K=H
    W_ptr,    # *ptr to [K, K] (process_weight, right-multiply)
    C_ptr,    # *ptr to [M, K]
    M,        # int: B*(T+I)
    K,        # int: H
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid over output tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * K + offs_k[None, :])
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # dtype follows A_ptr (float32 expected)

        # W tile as [BLOCK_K, BLOCK_N]: W[k, n]
        w_ptrs = W_ptr + (offs_k[:, None] * K + offs_n[None, :])
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)  # dtype follows W_ptr

        # Accumulate
        acc += tl.dot(a, w)

    # Write back to C
    c_ptrs = C_ptr + (offs_m[:, None] * K + offs_n[None, :])
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
    tl.store(c_ptrs, acc, mask=c_mask)


# Kernel 3: Split the concatenated processed tensor back into encoder and hidden streams.
@triton.jit
def _split_streams_kernel(
    C_ptr,           # *ptr to [B*(T+I), H]
    out_encoder_ptr, # *ptr to [B, T, H]
    out_hidden_ptr,  # *ptr to [B, I, H]
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    H: tl.constexpr,
    stride_c_m, stride_c_n,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_i, stride_h_h,
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)
    L = T + I
    # Loop over output rows m
    for m in range(0, B * L):
        # Map m to (batch, seq) and load
        seq = m % L
        # Load row from C
        c_row_ptr = C_ptr + m * stride_c_m
        h_idx = tl.arange(0, H)
        mask = h_idx < H
        vals = tl.load(c_row_ptr + h_idx * stride_c_n, mask=mask, other=0.0)

        # Write to appropriate output
        if seq < T:
            dst_ptr = out_encoder_ptr + b * stride_e_b + seq * stride_e_t
        else:
            dst_ptr = out_hidden_ptr + b * stride_h_b + (seq - T) * stride_h_i
        tl.store(dst_ptr + h_idx * stride_e_h, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, H]
        encoder_hidden_states: [B, T, H]
        process_weight: [H, H] (no bias)
        returns: (processed_encoder [B, T, H], processed_hidden [B, I, H])
        """
        assert hidden_states.ndim == 3 and encoder_hidden_states.ndim == 3 and process_weight.ndim == 2
        B = hidden_states.shape[0]
        I = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "hidden_dim must match"
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"

        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1) Concatenate in Triton
        out_cat = torch.empty((B, T + I, H), dtype=dtype, device=device)
        _concatenation_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            *encoder_hidden_states.stride(),
            *hidden_states.stride(),
            *out_cat.stride(),
        )

        # 2) GEMM in Triton: C = out_cat @ process_weight.T
        # out_cat is [B, T+I, H]; flatten to [M, K], M=B*(T+I), K=H
        M = B * (T + I)
        K = H
        C = torch.empty((M, K), dtype=torch.float32, device=device)  # compute in fp32

        # Ensure inputs to kernel are fp32
        A = out_cat.to(torch.float32).reshape(M, K).contiguous()
        W = process_weight.to(torch.float32).contiguous()  # [K, K]
        # Grid for 2D tiling
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _batched_matmul_kernel[grid](
            A, W, C,
            M, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split in Triton
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=device)
        _split_streams_kernel[(B,)](
            C, processed_encoder, processed_hidden,
            B, T, I, H,
            *C.stride(),
            *processed_encoder.stride(), *processed_hidden.stride(),
            BLOCK_H=128,
            num_warps=1, num_stages=1,
        )

        # Cast back to original dtype if needed
        if dtype != torch.float32:
            processed_encoder = processed_encoder.to(dtype)
            processed_hidden = processed_hidden.to(dtype)

        return processed_encoder, processed_hidden