import torch
import triton
import triton.language as tl


@triton.jit
def concatenate_seqs_kernel(
    enc_ptr,        # *ptr to encoder_hidden_states [B, T, H]
    hid_ptr,        # *ptr to hidden_states [B, I, H]
    out_ptr,        # *ptr to concatenated [B, S, H]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
    enc_stride0, enc_stride1, enc_stride2,
    hid_stride0, hid_stride1, hid_stride2,
    out_stride0, out_stride1, out_stride2,
    BLOCK_H: tl.constexpr,
):
    # grid: (B,)
    b = tl.program_id(0)
    # Each program handles one batch element and copies enc and hid into out
    # Iterate over H dimension in chunks
    for h0 in range(0, H, BLOCK_H):
        h_offsets = h0 + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H

        # Copy encoder rows [b, t, :] for t in [0, T)
        for t in range(0, T):
            base = b * enc_stride0 + t * enc_stride1
            enc_vals = tl.load(enc_ptr + base + h_offsets * enc_stride2, mask=mask_h, other=0.0)
            out_base = b * out_stride0 + t * out_stride1
            tl.store(out_ptr + out_base + h_offsets * out_stride2, enc_vals, mask=mask_h)

        # Copy hidden rows [b, i, :] for i in [0, I)
        for i in range(0, I):
            base = b * hid_stride0 + i * hid_stride1
            hid_vals = tl.load(hid_ptr + base + h_offsets * hid_stride2, mask=mask_h, other=0.0)
            # out position is at t = T + i
            out_base = b * out_stride0 + (T + i) * out_stride1
            tl.store(out_ptr + out_base + h_offsets * out_stride2, hid_vals, mask=mask_h)


@triton.jit
def matmul_seqs_kernel(
    A_ptr,  # *ptr to concatenated [B, S, H], we will use flattened rows [S, H]
    B_ptr,  # *ptr to process_weight.T [H, H]
    C_ptr,  # *ptr to processed [B, S, H], we will use flattened rows [S, H]
    Bsz: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    A_stride0, A_stride1,
    C_stride0, C_stride1,
    B_stride0, B_stride1,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # grid: (rows, tiles over N)
    row = tl.program_id(0)  # 0..(B*S - 1)
    tile_n = tl.program_id(1)
    n0 = tile_n * BLOCK_N
    n_offsets = n0 + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < H

    # Accumulator for this row segment
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < H

        # A[row, k] = A_ptr[row * A_stride0 + k * A_stride1]
        a_ptrs = A_ptr + row * A_stride0 + k_offsets * A_stride1
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)  # shape [BLOCK_K]

        # B[k, n] = B_ptr[k * B_stride0 + n * B_stride1]
        b_ptrs = B_ptr + k_offsets[:, None] * B_stride0 + n_offsets[None, :] * B_stride1
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)  # shape [BLOCK_K, BLOCK_N]

        # acc[n] += sum_k a[k] * b[k, n]
        acc += tl.sum(a[:, None] * b, axis=0)

    # Store result to C[row, n]
    c_ptrs = C_ptr + row * C_stride0 + n_offsets * C_stride1
    tl.store(c_ptrs, acc, mask=n_mask)


@triton.jit
def split_seqs_kernel(
    processed_ptr,  # *ptr to processed [B, S, H]
    out_e_ptr,      # *ptr to processed_encoder [B, T, H]
    out_i_ptr,      # *ptr to processed_hidden [B, I, H]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
    processed_stride0, processed_stride1, processed_stride2,
    out_e_stride0, out_e_stride1, out_e_stride2,
    out_i_stride0, out_i_stride1, out_i_stride2,
    BLOCK_H: tl.constexpr,
):
    # grid: (B,)
    b = tl.program_id(0)

    for h0 in range(0, H, BLOCK_H):
        h_offsets = h0 + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H

        # Copy first T rows -> encoder
        for t in range(0, T):
            base_in = b * processed_stride0 + t * processed_stride1
            vals = tl.load(processed_ptr + base_in + h_offsets * processed_stride2, mask=mask_h, other=0.0)
            base_out = b * out_e_stride0 + t * out_e_stride1
            tl.store(out_e_ptr + base_out + h_offsets * out_e_stride2, vals, mask=mask_h)

        # Copy remaining I rows -> hidden
        for i in range(0, I):
            base_in = b * processed_stride0 + (T + i) * processed_stride1
            vals = tl.load(processed_ptr + base_in + h_offsets * processed_stride2, mask=mask_h, other=0.0)
            base_out = b * out_i_stride0 + i * out_i_stride1
            tl.store(out_i_ptr + base_out + h_offsets * out_i_stride2, vals, mask=mask_h)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
          - Concatenate encoder_hidden_states and hidden_states along sequence dimension.
          - Apply linear projection via Triton matmul with process_weight.T.
          - Split back into encoder and hidden streams.
        Returns:
          processed_encoder: [batch, text_seq_len, hidden_dim]
          processed_hidden:  [batch, img_seq_len, hidden_dim]
        """
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA"
        assert encoder_hidden_states.dtype == torch.float32 and hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors"
        assert encoder_hidden_states.is_contiguous() and hidden_states.is_contiguous() and process_weight.is_contiguous(), "Tensors must be contiguous"

        B, T, H = encoder_hidden_states.shape
        _, I, H2 = hidden_states.shape
        assert H == H2, "hidden_dim must match between encoder_hidden_states and hidden_states"
        # process_weight: [H, H]
        assert process_weight.shape == (H, H), "process_weight must be [hidden_dim, hidden_dim]"

        # 1) Concatenate encoder and hidden into [B, S, H]
        S = T + I
        concatenated = torch.empty((B, S, H), device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)

        grid_concat = (B,)
        # Use a BLOCK_H that covers H; since H is often 1024 here, we can set BLOCK_H=128
        BLOCK_H = 128
        concatenate_seqs_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=2,
        )

        # 2) Compute processed = concatenated @ process_weight.T using Triton (per-row GEMM)
        # Concatenated has shape [B, S, H]; we will treat each row [S, H] and multiply by [H, H] -> [S, H].
        processed = torch.empty_like(concatenated)

        # Flattened views for kernels: A rows are [S, H], C rows are [S, H]
        grid_rows = B * S
        # 2D grid over (rows, tiles over N)
        BLOCK_N = 256
        BLOCK_K = 128
        grid_matmul = (grid_rows, triton.cdiv(H, BLOCK_N))
        matmul_seqs_kernel[grid_matmul](
            concatenated, process_weight.t(), processed,
            B, S, H,
            concatenated.stride(0), concatenated.stride(1),
            processed.stride(0), processed.stride(1),
            process_weight.t().stride(0), process_weight.t().stride(1),
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=3,
        )

        # 3) Split into encoder and hidden outputs
        processed_encoder = torch.empty((B, T, H), device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
        processed_hidden = torch.empty((B, I, H), device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)

        grid_split = (B,)
        split_seqs_kernel[grid_split](
            processed, processed_encoder, processed_hidden,
            B, T, I, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=2,
        )

        return processed_encoder, processed_hidden