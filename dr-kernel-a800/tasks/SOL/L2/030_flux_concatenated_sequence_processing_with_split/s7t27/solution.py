import torch
import triton
import triton.language as tl


@triton.jit
def matmul_fused_kernel(
    e_ptr, i_ptr, w_ptr, y_ptr,
    B, T, I, H,
    stride_eb, stride_et, stride_eh,
    stride_ib, stride_it, stride_ih,
    stride_wk, stride_wn,
    stride_yb, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs: batch, row tile, col tile
    b = tl.program_id(0)
    pid_m = tl.program_id(1)  # tile over M = T + I
    pid_n = tl.program_id(2)  # tile over H

    # Offsets for output tiles
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for boundaries
    m_mask = m_offsets < (T + I)
    n_mask = n_offsets < H

    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension (hidden dim)
    k0 = 0
    while k0 < H:
        # Determine source: encoder rows if k0 < T, else image rows at (k0 - T)
        from_encoder = k0 < T

        # Build pointers for e and i row blocks
        # We need rows corresponding to output row indices m_offsets.
        # For each output row p, source row index is p if p < T, else p - T. But since we're summing over k,
        # the source row is determined by from_encoder and k offset.
        # In Triton, we can't vectorize arbitrary branching across lanes easily here; instead we load from e and i
        # using k as the row index: for k < T, use e[b, k, n]; for k >= T, use i[b, k-T, n].
        # Load e rows for k < T and i rows for k >= T (unmasked) and combine using from_encoder flag.

        # Load W tile [BLOCK_K, BLOCK_N]
        k_vec = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_vec < H
        # When loading W, we only need indices within H. Use masks.

        # We'll iterate over k in steps of BLOCK_K:
        k_sub0 = 0
        while k_sub0 < H:
            k_vec = k_sub0 + tl.arange(0, BLOCK_K)
            k_mask = k_vec < H

            # Load W tile: W is [H, H]
            w_rows = k_vec[:, None]  # along K
            w_cols = n_offsets[None, :]  # along N
            w_tile = tl.load(
                w_ptr + w_rows * stride_wk + w_cols * stride_wn,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0
            )

            # Load X rows based on from_encoder flag:
            # We need X rows for each output row m_offsets:
            # If from_encoder: X_row = e[b, k_vec, n]; else X_row = i[b, k_vec-T, n]
            # However, we need to match X_cat concatenation: first T rows from e, next I rows from i.
            # The output row m corresponds to source row m if m < T; else source row m - T from i.
            # We can compute this per lane:
            # For each k in k_vec, determine from_encoder; then for each m lane, source_row = m if m < T else m - T
            # We'll do a simple approach: two separate loads and select, but Triton allows per-lane operations.

            # Build source row indices for each m lane:
            # src_row_e = tl.where(m_offsets[:, None] < T, m_offsets[:, None], m_offsets[:, None] - T)
            # src_row_e is invalid when m_offsets >= T; we handle by selecting i. We can compute per k whether to use e or i.

            # Compute source row for each output row m:
            # src_row_e = m_offsets[:, None] if m_offsets < T else m_offsets - T
            # To implement, we'll load from e when from_encoder is True, else from i.
            # from_encoder is a scalar (k_sub0 < T); but k_vec may exceed T; we need to mask by k_mask.

            # We'll load X rows using a combined approach: for each k in k_vec, use e if k < T, else i. We do this by loading both and selecting based on from_encoder.
            # However, Triton does not allow dynamic selection per k across lanes cleanly in this form. Instead, we'll load from e when from_encoder, else from i, by masking and tl.where.

            # Prepare src_row_e and src_row_i:
            # src_row_e = m_offsets[:, None]  # shape [BLOCK_M, 1], broadcast
            # src_row_i = m_offsets[:, None] - T

            # Build base row pointers:
            # For e: e_row = b * stride_eb + src_row_e * stride_et + k_vec[None, :] * stride_eh
            # For i: i_row = b * stride_ib + (src_row_i) * stride_it + (k_vec[None, :] - T) * stride_ih

            # We'll do this per k lane by looping k scalar within this tile. Triton allows scalar-loop over k_vec entries.

            # For Triton, we implement simple scalar-loop over k within k_sub0..H-1:
            k_idx = k_sub0
            while k_idx < H:
                # Scalar k
                # Determine from_encoder flag for this k
                use_e = k_idx < T
                # Compute src_row for each m lane: if m < T, src_row = k_idx; else src_row = k_idx - T
                src_row_e = m_offsets[:, None]  # [BLOCK_M, 1]
                src_row_i = m_offsets[:, None] - T  # [BLOCK_M, 1]

                # Load X_row scalar for each m:
                # X_row_e = e[b, src_row_e, n_offsets]
                # X_row_i = i[b, src_row_i, n_offsets] if use_e else i[b, src_row_e, n_offsets] (but src_row_e would be negative if use_e False; we need correct mapping)
                # The correct mapping is: if use_e: src_row = k_idx; else src_row = k_idx - T.
                # We'll load from e when use_e, else from i with src_row = k_idx - T. But we need to avoid negative indices; since k_idx >= T when use_e=False, k_idx - T >= 0.
                # Load from e:
                e_row = b * stride_eb + k_idx * stride_et
                e_cols = n_offsets[None, :]
                e_vals = tl.load(
                    e_ptr + e_row + e_cols * stride_eh,
                    mask=m_mask[:, None] & n_mask[None, :],
                    other=0.0
                )

                # Load from i:
                i_row = b * stride_ib + (k_idx - T) * stride_it
                i_vals = tl.load(
                    i_ptr + i_row + n_offsets[None, :] * stride_ih,
                    mask=m_mask[:, None] & n_mask[None, :],
                    other=0.0
                )

                # Select X_vals based on use_e (scalar)
                x_vals = tl.where(use_e, e_vals, i_vals)

                # Accumulate acc += x_vals * w_col for this k
                # w_col for this k is w_ptr[k_idx, n_offsets]
                w_col = tl.load(
                    w_ptr + k_idx * stride_wk + n_offsets[None, :] * stride_wn,
                    mask=n_mask[None, :],
                    other=0.0
                )
                acc += x_vals * w_col  # elementwise multiply and accumulate
                k_idx += 1

            k_sub0 += BLOCK_K

    # Store acc into Y for this tile
    y_row = b * stride_yb
    y_m_off = m_offsets[:, None] * stride_ym
    y_n_off = n_offsets[None, :] * stride_yn
    y_ptrs = y_ptr + y_row + y_m_off + y_n_off
    tl.store(y_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation: computes Y = cat([encoder_hidden_states, hidden_states], dim=1) @ process_weight
        Returns (processed_encoder_hidden_states, processed_hidden_states) corresponding to the split along sequence.
        """
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA device."

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        M = T + I

        # Make sure inputs are contiguous and float32
        e = encoder_hidden_states.contiguous()
        i = hidden_states.contiguous()
        w = process_weight.contiguous()

        e32 = e.float()
        i32 = i.float()
        w32 = w.float()

        # Allocate output
        Y = torch.empty((B, M, H), device=e32.device, dtype=torch.float32)

        # Choose tile sizes; these are reasonable defaults for typical H up to 1024
        BLOCK_M = 128 if M >= 128 else (64 if M >= 64 else 32)
        BLOCK_N = 128 if H >= 128 else (64 if H >= 64 else 32)
        BLOCK_K = 64 if H >= 64 else (32 if H >= 32 else 16)

        # Launch fused matmul kernel with 3D grid: (batch, tiles over M, tiles over H)
        grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        matmul_fused_kernel[grid](
            e32, i32, w32, Y,
            B, T, I, H,
            e32.stride(0), e32.stride(1), e32.stride(2),
            i32.stride(0), i32.stride(1), i32.stride(2),
            w32.stride(0), w32.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3
        )

        # Split streams
        processed_encoder = Y[:, :T, :]
        processed_hidden = Y[:, T:, :]

        # Cast back to original dtype
        processed_encoder = processed_encoder.to(e.dtype)
        processed_hidden = processed_hidden.to(e.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
