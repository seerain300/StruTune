import torch

# Triton kernels
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _concatenate_sequences_kernel(
        encoder_ptr,        # *ptr to [B, T, H]
        hidden_ptr,         # *ptr to [B, I, H]
        out_ptr,            # *ptr to [B, T+I, H]
        B: tl.constexpr,
        T: tl.constexpr,
        I: tl.constexpr,
        H: tl.constexpr,
        stride_e_b, stride_e_t, stride_e_h,
        stride_h_b, stride_h_i, stride_h_h,
        stride_o_b, stride_o_l, stride_o_h,
        BLOCK_H: tl.constexpr = 128,
    ):
        # one program per batch
        b = tl.program_id(0)
        total_seq = T + I

        # iterate over sequence positions
        l = 0
        while l < total_seq:
            # Determine which source tensor to read from
            is_encoder = l < T
            # Compute pointers
            if is_encoder:
                e_idx = b * stride_e_b + l * stride_e_t
                # Load entire hidden dimension for this row
                h_offs = tl.arange(0, BLOCK_H)
                mask_h = h_offs < H
                vals = tl.load(encoder_ptr + e_idx + h_offs * stride_e_h, mask=mask_h, other=0.0)
            else:
                h_offs = tl.arange(0, BLOCK_H)
                mask_h = h_offs < H
                vals = tl.load(hidden_ptr + b * stride_h_b + (l - T) * stride_h_i + h_offs * stride_h_h, mask=mask_h, other=0.0)

            # Store to output [B, T+I, H]
            o_idx = b * stride_o_b + l * stride_o_l
            tl.store(out_ptr + o_idx + h_offs * stride_o_h, vals, mask=mask_h)
            l += 1

    @triton.jit
    def _batched_matmul_right_kernel(
        A_ptr,      # *ptr to [M, K], where M = B*(T+I), K = H
        W_ptr,      # *ptr to [K, K] (process_weight, right-multiply by W.T)
        C_ptr,      # *ptr to [M, K] (output)
        M: tl.constexpr,  # total rows = B*(T+I)
        K: tl.constexpr,  # hidden_dim
        stride_am, stride_ak,   # strides for A: (row, col)
        stride_wk, stride_wn,   # strides for W: (row, col) == (k, n) since W is [K, K]
        stride_cm, stride_cn,   # strides for C: (row, col)
        BLOCK_M: tl.constexpr = 64,
        BLOCK_N: tl.constexpr = 64,
        BLOCK_K: tl.constexpr = 32,
    ):
        # 2D launch grid: programs over output tiles (m, n)
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        m0 = pid_m * BLOCK_M
        n0 = pid_n * BLOCK_N

        m_offs = m0 + tl.arange(0, BLOCK_M)
        n_offs = n0 + tl.arange(0, BLOCK_N)

        # Accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Loop over K in chunks
        for k0 in range(0, K, BLOCK_K):
            k_offs = k0 + tl.arange(0, BLOCK_K)

            # Load A tile: [BLOCK_M, BLOCK_K]
            a_ptrs = A_ptr + m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak
            a_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
            a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

            # Load W^T tile: we want [BLOCK_K, BLOCK_N] which is W[k, n] for k in K-chunk, n in N-chunk
            w_ptrs = W_ptr + k_offs[:, None] * stride_wk + n_offs[None, :] * stride_wn
            w_mask = (k_offs[:, None] < K) & (n_offs[None, :] < K)
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

            # Accumulate
            acc += tl.dot(a_tile, w_tile)

        # Store result
        c_ptrs = C_ptr + m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn
        c_mask = (m_offs[:, None] < M) & (n_offs[None, :] < K)
        tl.store(c_ptrs, acc, mask=c_mask)

    @triton.jit
    def _split_streams_kernel(
        C_ptr,            # *ptr to [B*(T+I), H]
        out_encoder_ptr,  # *ptr to [B, T, H]
        out_hidden_ptr,   # *ptr to [B, I, H]
        B: tl.constexpr,
        T: tl.constexpr,
        I: tl.constexpr,
        H: tl.constexpr,
        stride_c_row, stride_c_col,        # C strides (row=m, col=n)
        stride_e_b, stride_e_t, stride_e_h,
        stride_h_b, stride_h_i, stride_h_h,
        BLOCK_H: tl.constexpr = 128,
    ):
        # one program per batch
        b = tl.program_id(0)

        # First stream: encoder (first T rows in C)
        for t in range(0, T):
            m = b * (T + I) + t
            # copy each hidden dim column
            for n in range(0, H):
                val = tl.load(C_ptr + m * stride_c_row + n * stride_c_col)
                tl.store(out_encoder_ptr + b * stride_e_b + t * stride_e_t + n * stride_e_h, val)

        # Second stream: hidden (next I rows in C)
        for i in range(0, I):
            m = b * (T + I) + T + i
            for n in range(0, H):
                val = tl.load(C_ptr + m * stride_c_row + n * stride_c_col)
                tl.store(out_hidden_ptr + b * stride_h_b + i * stride_h_i + n * stride_h_h, val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure device and dtype compatibility
        device = hidden_states.device
        dtype = hidden_states.dtype
        assert process_weight.device == device, "process_weight must be on the same device as inputs"
        assert process_weight.dtype == dtype, "process_weight dtype should match input dtype"

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "hidden_dim must match between encoder and image inputs"
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [hidden_dim, hidden_dim]"

        # Allocate concatenated tensor [B, T+I, H] as float32 for robustness, then cast at the end if needed
        out_cat = torch.empty((B, T + I, H), dtype=dtype, device=device)

        # Launch concatenation kernel: one program per batch
        _concatenate_sequences_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            *encoder_hidden_states.stride(),
            *hidden_states.stride(),
            *out_cat.stride(),
            BLOCK_H=128,
            num_warps=1,
            num_stages=1,
        )

        # Make A = [M, K] where M = B*(T+I), K = H
        A = out_cat  # already [B, T+I, H]
        M = B * (T + I)
        K = H

        # Allocate C as float32 for accumulation
        C = torch.empty((M, K), dtype=torch.float32, device=device)

        # Launch batched matmul: grid over output tiles
        grid_m = (M + 63) // 64
        grid_n = (K + 63) // 64
        _batched_matmul_right_kernel[(grid_m, grid_n)](
            A, process_weight, C,
            M, K,
            A.stride(0), A.stride(2),         # A is [B, T+I, H], strides: row=(T+I)*H, col=1 (contiguous hidden dim)
            process_weight.stride(0), process_weight.stride(1),  # W is [K, K], strides: (k, n)
            C.stride(0), C.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2,
        )

        # Allocate outputs
        processed_encoder = torch.empty((B, T, H), dtype=dtype, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=dtype, device=device)

        # Launch split kernel: one program per batch
        _split_streams_kernel[(B,)](
            C, processed_encoder, processed_hidden,
            B, T, I, H,
            C.stride(0), C.stride(1),
            *processed_encoder.stride(), *processed_hidden.stride(),
            BLOCK_H=128,
            num_warps=1, num_stages=1,
        )

        return processed_encoder, processed_hidden