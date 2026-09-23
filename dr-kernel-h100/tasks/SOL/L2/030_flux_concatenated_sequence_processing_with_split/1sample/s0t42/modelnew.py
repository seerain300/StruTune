import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: concatenate [B, T, H] and [B, I, H] into [B, T+I, H]
# One program per batch
if TRITON_AVAILABLE:
    @triton.jit
    def _concat_encoder_hidden_kernel(
        e_ptr,            # *const T [B, T, H]
        h_ptr,            # *const T [B, I, H]
        out_ptr,          # *T [B, T+I, H]
        B: tl.constexpr,  # batch size
        T: tl.constexpr,  # text sequence length
        I: tl.constexpr,  # image sequence length
        H: tl.constexpr,  # hidden dim
        stride_e_b, stride_e_t, stride_e_h,
        stride_h_b, stride_h_i, stride_h_h,
        stride_out_b, stride_out_l, stride_out_h,
        BLOCK_H: tl.constexpr,
    ):
        b = tl.program_id(0)
        # Guard batch
        if b >= B:
            return

        # Loop over concatenated sequence length
        L = T + I
        for l in range(0, L):
            if l < T:
                row_e = e_ptr + b * stride_e_b + l * stride_e_t
            else:
                row_e = h_ptr + b * stride_h_b + (l - T) * stride_h_i
            # Store to output
            out_row = out_ptr + b * stride_out_b + l * stride_out_l
            # Vector of H dimension
            h_offsets = tl.arange(0, BLOCK_H)
            mask = h_offsets < H
            vals = tl.load(row_e + h_offsets * stride_e_h, mask=mask, other=0.0)
            tl.store(out_row + h_offsets * stride_out_h, vals, mask=mask)


# Triton kernel: 2D-tiled batched matmul over A[M,K] and W[K,K]^T -> C[M,K]
# Grid: (grid_m, grid_n) tiles over M and K output dimensions
if TRITON_AVAILABLE:
    @triton.jit
    def _batched_matmul_kernel(
        A_ptr,   # *const float [M, K], A = out_cat flattened
        W_ptr,   # *const float [K, K], W is process_weight, right-multiply W.T
        C_ptr,   # *float [M, K], output
        M: tl.constexpr,  # total rows = B*(T+I)
        K: tl.constexpr,  # hidden dim
        stride_A_m, stride_A_k,
        stride_W_k, stride_W_j,
        stride_C_m, stride_C_k,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        # Accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Reduce over K
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)

            # Load A tile: [BLOCK_M, BLOCK_K]
            a_ptrs = A_ptr + (offs_m[:, None] * stride_A_m) + (offs_k[None, :] * stride_A_k)
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # dtype determined by A_ptr (float32)

            # Load W^T tile: we want [BLOCK_K, BLOCK_N] corresponding to W[j, k], with j in [k0+{0..BLOCK_K-1}], k in [offs_n]
            # W has shape [K, K]. Access w[j, offs_n] for j in offs_k.
            w_ptrs = W_ptr + (offs_k[:, None] * stride_W_k) + (offs_n[None, :] * stride_W_j)
            w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < K)
            w = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

            # Dot: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
            acc += tl.dot(a, w)

        # Store results
        c_ptrs = C_ptr + (offs_m[:, None] * stride_C_m) + (offs_n[None, :] * stride_C_k)
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
        tl.store(c_ptrs, acc, mask=c_mask)


# Triton kernel: split C[M,K] into processed_encoder [B,T,K] and processed_hidden [B,I,K]
# One program per batch, loops over rows m and writes to the correct stream
if TRITON_AVAILABLE:
    @triton.jit
    def _split_streams_kernel(
        C_ptr,            # *const float [M, K]
        out_e_ptr,        # *float [B, T, K]
        out_h_ptr,        # *float [B, I, K]
        B: tl.constexpr,
        T: tl.constexpr,
        I: tl.constexpr,
        K: tl.constexpr,                      # hidden dim
        stride_C_m, stride_C_k,
        stride_e_b, stride_e_t, stride_e_k,
        stride_h_b, stride_h_i, stride_h_k,
        BLOCK_H: tl.constexpr,
    ):
        b = tl.program_id(0)
        if b >= B:
            return
        L = T + I
        # Iterate over all rows m = 0..M-1
        for m in range(0, M):
            # Map m to (b, l)
            b_idx = m // L
            l_idx = m % L
            # Load row from C
            c_row = C_ptr + m * stride_C_m
            h_offsets = tl.arange(0, BLOCK_H)
            mask = h_offsets < K
            vals = tl.load(c_row + h_offsets * stride_C_k, mask=mask, other=0.0)

            if l_idx < T:
                dest = out_e_ptr + b * stride_e_b + l_idx * stride_e_t
            else:
                dest = out_h_ptr + b * stride_h_b + (l_idx - T) * stride_h_i
            tl.store(dest + h_offsets * stride_e_k, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension
        - Apply linear projection to the concatenated sequence using a Triton matmul kernel
        - Split back into separate encoder and image streams using Triton
        Returns: (processed_encoder_hidden_states, processed_hidden_states)
        """
        assert TRITON_AVAILABLE, "Triton is not available. Please install triton to run ModelNew."
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Ensure all tensors are on the same device and dtype
        encoder_hidden_states = encoder_hidden_states.to(device=device, dtype=dtype)
        hidden_states = hidden_states.to(device=device, dtype=dtype)
        process_weight = process_weight.to(device=device, dtype=torch.float32)  # compute in fp32 for stability

        # Allocate concatenated output [B, T+I, H]
        out_cat = torch.empty((B, T + I, H), dtype=dtype, device=device)

        # Launch concatenation kernel: one program per batch
        # Pass strides
        stride_e_b, stride_e_t, stride_e_h = encoder_hidden_states.stride()
        stride_h_b, stride_h_i, stride_h_h = hidden_states.stride()
        stride_out_b, stride_out_l, stride_out_h = out_cat.stride()

        # Choose BLOCK_H to cover H (power of two up to 128 is fine)
        # Use next power of two for simple masking; but we can just set BLOCK_H = 128 and mask by H.
        BLOCK_H = 128

        _concat_encoder_hidden_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            stride_e_b, stride_e_t, stride_e_h,
            stride_h_b, stride_h_i, stride_h_h,
            stride_out_b, stride_out_l, stride_out_h,
            BLOCK_H=BLOCK_H,
            num_warps=1, num_stages=1,
        )

        # Prepare A = out_cat flattened to [M, K]
        A = out_cat.contiguous()  # ensure contiguous for simple strides
        M = B * (T + I)
        K = H
        stride_A_m, stride_A_k = A.stride(0), A.stride(1)

        # Prepare W as [K, K] (process_weight is [H, H], but we need W^T to right-multiply; we load tiles as needed)
        # W is process_weight, we will use it directly as [K, K] logically by indexing W[j, k]. For Triton, we pass W as is.
        W = process_weight  # [H, H], ensure contiguous
        W = W.contiguous()
        stride_W_k, stride_W_j = W.stride(0), W.stride(1)  # [K, K], here j is hidden_dim, k is output dim (both H)

        # Allocate C = [M, K], float32 accumulation
        C = torch.empty((M, K), dtype=torch.float32, device=device)
        stride_C_m, stride_C_k = C.stride(0), C.stride(1)

        # Choose tiling parameters
        BLOCK_M = 64   # tile rows of A
        BLOCK_N = 64   # tile output columns
        BLOCK_K = 32   # reduction tile over K

        grid_m = (M + BLOCK_M - 1) // BLOCK_M
        grid_n = (K + BLOCK_N - 1) // BLOCK_N

        _batched_matmul_kernel[(grid_m, grid_n)](
            A, W, C,
            M, K,
            stride_A_m, stride_A_k,
            stride_W_k, stride_W_j,
            stride_C_m, stride_C_k,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Split into [B, T, K] and [B, I, K]
        processed_encoder = torch.empty((B, T, K), dtype=dtype, device=device)
        processed_hidden = torch.empty((B, I, K), dtype=dtype, device=device)

        stride_e_b, stride_e_t, stride_e_k = processed_encoder.stride()
        stride_h_b, stride_h_i, stride_h_k = processed_hidden.stride()

        _split_streams_kernel[(B,)](
            C, processed_encoder, processed_hidden,
            B, T, I, K,
            stride_C_m, stride_C_k,
            stride_e_b, stride_e_t, stride_e_k,
            stride_h_b, stride_h_i, stride_h_k,
            BLOCK_H=128,  # K is typically <= 1024; 128 works as a safe default
            num_warps=1, num_stages=1,
        )

        return processed_encoder, processed_hidden