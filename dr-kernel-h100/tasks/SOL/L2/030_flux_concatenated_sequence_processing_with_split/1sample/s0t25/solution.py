import triton
import triton.language as tl

# Triton kernel: concatenate encoder_hidden_states and hidden_states along sequence dim
# Inputs:
#   enc: [B, T, H], row-major, contiguous
#   hid: [B, I, H], row-major, contiguous
# Output:
#   out_cat: [B, T+I, H], row-major, contiguous
@triton.jit
def _concat_sequences_kernel(
    enc_ptr, hid_ptr, out_ptr,
    B, T, I, H,
    enc_stride_b, enc_stride_t, enc_stride_h,
    hid_stride_b, hid_stride_i, hid_stride_h,
    out_stride_b, out_stride_s, out_stride_h,
    BLOCK_S: tl.constexpr,  # block for sequence (T+I)
):
    b = tl.program_id(0)  # one program per batch
    # Offsets along the concatenated sequence dimension
    s = tl.arange(0, BLOCK_S)
    total = T + I
    # Loop over sequence chunks
    for off in range(0, total, BLOCK_S):
        s_idx = off + s
        mask_s = s_idx < total
        # Determine source: first T rows from enc, remaining from hid
        is_enc = s_idx < T
        # Compute per-element pointers for enc and hid
        # enc: out[b, s_idx, h] = enc[b, s_idx, h] if s_idx < T else 0
        # hid: out[b, s_idx, h] = hid[b, s_idx - T, h] if s_idx >= T else 0
        h = tl.arange(0, H)
        # Build mask for enc/hid loads
        mask = mask_s[:, None]  # broadcast over H
        # Load enc rows
        enc_ptrs = enc_ptr + b * enc_stride_b + s_idx[:, None] * enc_stride_t + h[None, :] * enc_stride_h
        enc_vals = tl.load(enc_ptrs, mask=mask & is_enc[:, None], other=0.0)
        # Load hid rows (only when s_idx >= T)
        hid_ptrs = hid_ptr + b * hid_stride_b + (s_idx[:, None] - T) * hid_stride_i + h[None, :] * hid_stride_h
        hid_vals = tl.load(hid_ptrs, mask=mask & (~is_enc)[:, None], other=0.0)
        # Combine
        vals = enc_vals + hid_vals
        # Store to out_cat
        out_ptrs = out_ptr + b * out_stride_b + s_idx[:, None] * out_stride_s + h[None, :] * out_stride_h
        tl.store(out_ptrs, vals, mask=mask)

# Triton kernel: split concatenated rows back into encoder and hidden streams
# Input:
#   cat: [B*(T+I), H], contiguous (row-major). We pass M rows where M = B*(T+I)
# Output:
#   encoder_out: [B, T, H], contiguous
#   hidden_out: [B, I, H], contiguous
@triton.jit
def _split_streams_kernel(
    cat_ptr, encoder_ptr, hidden_ptr,
    B, T, I, H,
    cat_stride_m, cat_stride_k,
    enc_stride_b, enc_stride_t, enc_stride_h,
    hid_stride_b, hid_stride_i, hid_stride_h,
    BLOCK_K: tl.constexpr,  # tile over H (columns)
):
    # One program per batch
    b = tl.program_id(0)
    # We will write T rows to encoder and I rows to hidden
    # First write encoder rows: rows [0..T-1] of cat
    for t in range(0, T):
        # Row pointer: cat[b*(T+I) + t, :]
        row = b * (T + I) + t
        k = tl.arange(0, BLOCK_K)
        mask_k = k < H
        ptrs = cat_ptr + row * cat_stride_m + k * cat_stride_k
        vals = tl.load(ptrs, mask=mask_k, other=0.0)
        enc_ptrs = encoder_ptr + b * enc_stride_b + t * enc_stride_t + k * enc_stride_h
        tl.store(enc_ptrs, vals, mask=mask_k)
    # Then write hidden rows: rows [T..T+I-1] of cat
    for i in range(0, I):
        row = b * (T + I) + T + i
        k = tl.arange(0, BLOCK_K)
        mask_k = k < H
        ptrs = cat_ptr + row * cat_stride_m + k * cat_stride_k
        vals = tl.load(ptrs, mask=mask_k, other=0.0)
        hid_ptrs = hidden_ptr + b * hid_stride_b + i * hid_stride_i + k * hid_stride_h
        tl.store(hid_ptrs, vals, mask=mask_k)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        - Concatenate sequences along dim=1 in Triton
        - Apply linear projection via torch.matmul (right-multiply by process_weight.T)
        - Split results back into encoder and hidden streams in Triton

        Args:
            hidden_states: [B, I, H]
            encoder_hidden_states: [B, T, H]
            process_weight: [H, H]
        Returns:
            processed_encoder: [B, T, H]
            processed_hidden: [B, I, H]
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D tensors"
        assert process_weight.dim() == 2, "process_weight must be 2D [H, H]"
        B, I, H = hidden_states.shape
        T = encoder_hidden_states.shape[1]
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Ensure inputs are contiguous and same dtype/device
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        W = process_weight.contiguous()

        # 1) Concatenate in Triton: out_cat [B, T+I, H]
        total = T + I
        out_cat = torch.empty((B, total, H), dtype=dtype, device=device)
        # Launch: one program per batch
        # Choose BLOCK_S to cover total seq length; use 1024 as safe default
        _concat_sequences_kernel[(B,)](
            enc, hid, out_cat,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            hid.stride(0), hid.stride(1), hid.stride(2),
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            BLOCK_S=1024,
            num_warps=4,
            num_stages=2,
        )

        # 2) GEMM: processed = out_cat @ W.T  -> shape [B, T+I, H]
        # Use torch.matmul for robustness and speed
        out = torch.matmul(out_cat, W.t())

        # 3) Split in Triton into [B, T, H] and [B, I, H]
        encoder_out = torch.empty((B, T, H), dtype=dtype, device=device)
        hidden_out = torch.empty((B, I, H), dtype=dtype, device=device)

        M = B * (T + I)
        _split_streams_kernel[(B,)](
            out,
            encoder_out, hidden_out,
            B, T, I, H,
            out.stride(0), out.stride(1),
            encoder_out.stride(0), encoder_out.stride(1), encoder_out.stride(2),
            hidden_out.stride(0), hidden_out.stride(1), hidden_out.stride(2),
            BLOCK_K=1024,  # tile over H; H<=1024 in provided workloads
            num_warps=4,
            num_stages=2,
        )

        return encoder_out, hidden_out


def run(*args):
    return ModelNew()(*args)
