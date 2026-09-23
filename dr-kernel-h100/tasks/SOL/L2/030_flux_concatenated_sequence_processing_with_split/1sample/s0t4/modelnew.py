import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_seq_dim_kernel(
    enc_ptr,        # *ptr to [B, T, H]
    hid_ptr,        # *ptr to [B, I, H]
    out_ptr,        # *ptr to [B, T+I, H]
    B, T, I, H,     # ints
    enc_s0, enc_s1, enc_s2,  # strides for enc: (B, T, H)
    hid_s0, hid_s1, hid_s2,  # strides for hid: (B, I, H)
    out_s0, out_s1, out_s2,  # strides for out: (B, T+I, H)
    BLOCK_L: tl.constexpr,
):
    b = tl.program_id(0)  # one program per batch
    # Loop over concatenated sequence length
    for l in range(0, T + I):
        mask = l < (T + I)
        if l < T:
            # Load from encoder_hidden_states
            enc_row = enc_ptr + b * enc_s0 + l * enc_s1
            vals = tl.load(enc_row + tl.arange(0, H) * enc_s2, mask=mask, other=0.0)
            out_row = out_ptr + b * out_s0 + l * out_s1
            tl.store(out_row + tl.arange(0, H) * out_s2, vals, mask=mask)
        else:
            # Load from hidden_states
            i = l - T
            hid_row = hid_ptr + b * hid_s0 + i * hid_s1
            vals = tl.load(hid_row + tl.arange(0, H) * hid_s2, mask=mask, other=0.0)
            out_row = out_ptr + b * out_s0 + l * out_s1
            tl.store(out_row + tl.arange(0, H) * out_s2, vals, mask=mask)


@triton.jit
def _batched_gemm_right_kernel(
    A_ptr,           # *ptr to [M, K], M = B*(T+I), K = H
    W_ptr,           # *ptr to [K, K]
    C_ptr,           # *ptr to [M, K] output
    M, K,            # ints: M rows, K cols
    A_s0, A_s1,      # strides for A: (M, K)
    W_s0, W_s1,      # strides for W: (K, K)
    C_s0, C_s1,      # strides for C: (M, K)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D launch: grid = (grid_m, grid_n)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + rows[:, None] * A_s0 + ks[None, :] * A_s1
        a_mask = (rows[:, None] < M) & (ks[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K], dtype follows input

        # Load W tile as [BLOCK_K, BLOCK_N]: W[k, cols]
        w_ptrs = W_ptr + ks[:, None] * W_s0 + cols[None, :] * W_s1
        w_mask = (ks[:, None] < K) & (cols[None, :] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N], dtype follows input

        # Accumulate
        acc += tl.dot(a.to(tl.float32), w.to(tl.float32))

    # Write back to C
    c_ptrs = C_ptr + rows[:, None] * C_s0 + cols[None, :] * C_s1
    c_mask = (rows[:, None] < M) & (cols[None, :] < K)
    # Cast back to original dtype of C (assuming same as A/W)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _split_streams_kernel(
    C_ptr,           # *ptr to [M, K], M = B*(T+I), K = H
    out_enc_ptr,     # *ptr to [B, T, K]
    out_hid_ptr,     # *ptr to [B, I, K]
    M, T, I, K,      # ints
    C_s0, C_s1,      # strides for C: (M, K)
    enc_s0, enc_s1, enc_s2,  # strides for out_enc: (B, T, K)
    hid_s0, hid_s1, hid_s2,  # strides for out_hid: (B, I, K)
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)  # one program per batch
    # First, write encoder part: rows [0, T)
    # For each l in [0, T)
    for l in range(0, T):
        row_index = b * (T + I) + l
        # Load row from C
        c_row_ptrs = C_ptr + row_index * C_s0 + tl.arange(0, K) * C_s1
        c_mask = True  # row exists, K is valid
        vals = tl.load(c_row_ptrs, mask=c_mask, other=0.0)
        # Store into out_enc[b, l, :]
        out_enc_row = out_enc_ptr + b * enc_s0 + l * enc_s1
        tl.store(out_enc_row + tl.arange(0, K) * enc_s2, vals)

    # Then, write hidden part: rows [T, T+I)
    for l in range(0, I):
        row_index = b * (T + I) + (T + l)
        c_row_ptrs = C_ptr + row_index * C_s0 + tl.arange(0, K) * C_s1
        c_mask = True
        vals = tl.load(c_row_ptrs, mask=c_mask, other=0.0)
        # Store into out_hid[b, l, :]
        out_hid_row = out_hid_ptr + b * hid_s0 + l * hid_s1
        tl.store(out_hid_row + tl.arange(0, K) * hid_s2, vals)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, H]
        encoder_hidden_states: [B, T, H]
        process_weight: [H, H] (no bias), we multiply by W.T
        Returns:
        processed_encoder: [B, T, H]
        processed_hidden: [B, I, H]
        """
        B, I, H = hidden_states.shape
        B2, T, H2 = encoder_hidden_states.shape
        assert B == B2 and H == H2, "Batch and hidden_dim must match across inputs"
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"

        # Ensure inputs are on the same device and contiguous
        device = hidden_states.device
        out_cat = torch.empty((B, T + I, H), dtype=hidden_states.dtype, device=device)

        # 1) Concatenate sequences along the sequence dimension using Triton
        _concatenate_seq_dim_kernel[(B,)](
            encoder_hidden_states,
            hidden_states,
            out_cat,
            B, T, I, H,
            *encoder_hidden_states.stride(),  # enc_s0, enc_s1, enc_s2
            *hidden_states.stride(),          # hid_s0, hid_s1, hid_s2
            *out_cat.stride(),                # out_s0, out_s1, out_s2
            BLOCK_L=1024,  # big enough to cover T+I; masked by (l < T+I)
        )

        # 2) Batched GEMM: C = out_cat @ process_weight.T
        # out_cat shape [B, T+I, H] -> [M, K], A: [M, K], W: [K, K]
        M = B * (T + I)
        K = H
        A = out_cat  # [B, T+I, H]
        # W right-multiply: [K, K]
        # Make sure W is contiguous
        W = process_weight  # [H, H], right multiply
        C = torch.empty((M, K), dtype=torch.float32, device=device)  # compute in fp32 for stability

        # Launch GEMM Triton kernel
        # Choose reasonable tiles for typical sizes; masks handle edge cases.
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        grid_m = (M + BLOCK_M - 1) // BLOCK_M
        grid_n = (K + BLOCK_N - 1) // BLOCK_N

        _batched_gemm_right_kernel[(grid_m, grid_n)](
            A, W, C,
            M, K,
            *A.stride(),    # A strides: (M, K) if contiguous => (T+I*H, H), but Triton uses actual strides
            *W.stride(),    # W strides: (K, K)
            *C.stride(),    # C strides: (M, K)
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=2,
        )

        # 3) Split into encoder and hidden streams using Triton
        processed_encoder = torch.empty((B, T, K), dtype=torch.float32, device=device)  # K=H
        processed_hidden = torch.empty((B, I, K), dtype=torch.float32, device=device)

        _split_streams_kernel[(B,)](
            C,
            processed_encoder, processed_hidden,
            M, T, I, K,
            *C.stride(),
            *processed_encoder.stride(), *processed_hidden.stride(),
            BLOCK_K=1024,
            num_warps=1,
            num_stages=1,
        )

        # Return as requested (original hidden_dim stream length); K==H
        return processed_encoder, processed_hidden