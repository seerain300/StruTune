import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Triton kernel: concatenate along sequence dimension for a given batch
# Input: enc[b, t, h], hid[b, i, h] -> out[b, t+i, h]
@triton.jit
def _concatenator_kernel(
    enc_ptr, hid_ptr, out_ptr,
    B, T, I, H,
    enc_bs, enc_ts, enc_hs,
    hid_bs, hid_is, hid_hs,
    out_bs, out_ts, out_hs,
    BLOCK_T: tl.constexpr, BLOCK_I: tl.constexpr
):
    b = tl.program_id(0)
    # Pointers to the start of this batch
    enc_b_ptr = enc_ptr + b * enc_bs
    hid_b_ptr = hid_ptr + b * hid_bs
    out_b_ptr = out_ptr + b * out_bs

    # Loop over T and I with masks, write to out[b, :, :]
    # We tile T and I for better parallelism (though small here).
    for t_start in range(0, T, BLOCK_T):
        for i_start in range(0, I, BLOCK_I):
            # Compute vector of indices within tile
            t_idx = t_start + tl.arange(0, BLOCK_T)
            i_idx = i_start + tl.arange(0, BLOCK_I)

            # Masks for valid indices
            mask_t = t_idx < T
            mask_i = i_idx < I

            # Load encoder rows: shape [BLOCK_T, H]
            # We'll write them to out[b, t_idx, :]
            enc_rows = tl.load(
                enc_b_ptr + t_idx[:, None] * enc_ts + tl.arange(0, H)[None, :] * enc_hs,
                mask=mask_t[:, None],
                other=0.0
            )  # [BLOCK_T, H]

            # Load hidden rows: shape [BLOCK_I, H]
            hid_rows = tl.load(
                hid_b_ptr + i_idx[:, None] * hid_is + tl.arange(0, H)[None, :] * hid_hs,
                mask=mask_i[:, None],
                other=0.0
            )  # [BLOCK_I, H]

            # Store encoder rows to out[b, t_idx, :]
            tl.store(
                out_b_ptr + t_idx[:, None] * out_ts + tl.arange(0, H)[None, :] * out_hs,
                enc_rows,
                mask=mask_t[:, None]
            )

            # Store hidden rows to out[b, I + i_idx, :]
            # Destination column index = I + i_idx
            dest_l = I + i_idx[None, :]  # [1, BLOCK_I]
            tl.store(
                out_b_ptr + dest_l * out_ts + tl.arange(0, H)[None, :] * out_hs,
                hid_rows,
                mask=mask_i[None, :]
            )


# Triton kernel: split C of shape [B, T+I, H] into two outputs
# processed_encoder [B, T, H] and processed_hidden [B, I, H]
@triton.jit
def _split_streams_kernel(
    C_ptr, out_enc_ptr, out_hid_ptr,
    B, T, I, H,
    C_bs, C_ts, C_hs,
    enc_bs, enc_ts, enc_hs,
    hid_bs, hid_ts, hid_hs,
    BLOCK_H: tl.constexpr
):
    # One program per batch
    b = tl.program_id(0)
    C_b_ptr = C_ptr + b * C_bs
    enc_b_ptr = out_enc_ptr + b * enc_bs
    hid_b_ptr = out_hid_ptr + b * hid_bs

    # Copy first T rows: C[b, :, :]
    for t in range(0, T):
        C_row_ptr = C_b_ptr + t * C_ts
        enc_row_ptr = enc_b_ptr + t * enc_ts
        h_idx = tl.arange(0, H)
        val = tl.load(C_row_ptr + h_idx * C_hs)
        tl.store(enc_row_ptr + h_idx * enc_hs, val)

    # Copy next I rows starting from index I: C[b, I + :, :]
    for i in range(0, I):
        dest_l = I + i
        C_row_ptr = C_b_ptr + dest_l * C_ts
        hid_row_ptr = hid_b_ptr + i * hid_ts
        h_idx = tl.arange(0, H)
        val = tl.load(C_row_ptr + h_idx * C_hs)
        tl.store(hid_row_ptr + h_idx * hid_hs, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, H]
        encoder_hidden_states: [B, T, H]
        process_weight: [H, H] (no bias), right-multiply
        returns: (processed_encoder [B, T, H], processed_hidden [B, I, H])
        """
        assert hidden_states.ndim == 3 and encoder_hidden_states.ndim == 3, "Inputs must be 3D tensors"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "hidden_dim must match between inputs"
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Ensure contiguous inputs for Triton
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        W = process_weight.contiguous()

        # Output buffer for concatenated [B, T+I, H]
        # Use float32 for matmul compute; cast back at end if needed. Here we keep dtype same as inputs.
        # Note: we will use torch.matmul for the GEMM to ensure correctness.
        out_cat = torch.empty((B, T + I, H), dtype=dtype, device=device)

        # Launch Triton concatenation kernel: one program per batch
        # Choose small tiles for H, since H is typically modest; for larger H, loop over H vector.
        BLOCK_T = 128 if T >= 128 else 64
        BLOCK_I = 128 if I >= 128 else 64

        _concatenator_kernel[(B,)](
            enc, hid, out_cat,
            B, T, I, H,
            *enc.stride(), *hid.stride(), *out_cat.stride(),
            BLOCK_T=BLOCK_T, BLOCK_I=BLOCK_I,
            num_warps=1, num_stages=1
        )

        # Perform batched GEMM using torch (reliable). A is [B, T+I, H], W is [H, H], right-multiply.
        # A @ W.T -> [B, T+I, H]
        # Note: if H is small, this is fast; if large, torch.matmul is optimized.
        # Ensure dtype matches process_weight; compute in float32 if inputs are fp16/bf16 for stability.
        if out_cat.dtype in (torch.float16, torch.bfloat16):
            out_mat = torch.matmul(out_cat.float(), W.float())
        else:
            out_mat = torch.matmul(out_cat, W)

        # Triton split: split along sequence axis back into two streams.
        processed_encoder = torch.empty((B, T, H), dtype=out_mat.dtype, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=out_mat.dtype, device=device)

        # Ensure outputs are contiguous (we will write into them)
        processed_encoder = processed_encoder.contiguous()
        processed_hidden = processed_hidden.contiguous()

        # Split using Triton kernel
        _split_streams_kernel[(B,)](
            out_mat, processed_encoder, processed_hidden,
            B, T, I, H,
            *out_mat.stride(), *processed_encoder.stride(), *processed_hidden.stride(),
            BLOCK_H=256 if H >= 256 else 128,
            num_warps=1, num_stages=1
        )

        return processed_encoder, processed_hidden