import torch
import triton
import triton.language as tl

# Triton kernel: concatenate [B, T, H] and [B, I, H] into [B, T+I, H]
@triton.jit
def _concatenate_sequences_kernel(
    enc_ptr,            # *ptr to encoder_hidden_states [B, T, H]
    hid_ptr,            # *ptr to hidden_states [B, I, H]
    out_ptr,            # *ptr to output [B, T+I, H]
    B, T, I, H,
    enc_stride_b, enc_stride_t, enc_stride_h,
    hid_stride_b, hid_stride_i, hid_stride_h,
    out_stride_b, out_stride_l, out_stride_h,
    BLOCK_L: tl.constexpr,
):
    b = tl.program_id(0)  # one program per batch
    # Loop over the concatenated sequence length
    for l in range(0, T + I, BLOCK_L):
        l_idx = l + tl.arange(0, BLOCK_L)
        mask_l = l_idx < (T + I)
        # Determine source: l < T from encoder, else from hidden
        is_encoder = l_idx < T
        # Compute base offsets
        base_enc = b * enc_stride_b
        base_hid = b * hid_stride_b
        # Load from encoder if within T, else from hidden
        # Create pointers for enc and hid loads
        enc_ptrs = enc_ptr + base_enc + l_idx * enc_stride_t + tl.arange(0, H) * enc_stride_h
        hid_ptrs = hid_ptr + base_hid + (l_idx - T) * hid_stride_i + tl.arange(0, H) * hid_stride_h
        # Masked load: use is_encoder for selecting
        # We'll build a 2D pointer for load: [BLOCK_L, H]
        # For masked elements where is_encoder is False, we don't want to load; use zeros.
        # To simplify, we load for encoder positions and write zero for hid positions.
        # Note: Triton doesn't support conditional 2D selection easily, so we do per-l and assign via mask.
        # Implement by looping over j in H (small H, e.g., 64-256) and writing with mask based on is_encoder.
        # This keeps the kernel simple and robust.
        # We'll loop j over H using tl.arange(0, H) and combine masks.
        j = tl.arange(0, H)
        for jj in range(0, H):
            # For each jj, choose source based on is_encoder
            # Compute enc and hid values with masks
            enc_val = tl.load(enc_ptr + base_enc + (l_idx[:, None] * enc_stride_t + jj * enc_stride_h), mask=mask_l[:, None] & is_encoder[:, None], other=0.0)
            hid_val = tl.load(hid_ptr + base_hid + ((l_idx - T)[:, None] * hid_stride_i + jj * hid_stride_h), mask=mask_l[:, None] & (~is_encoder)[:, None], other=0.0)
            # Select source: encoder if is_encoder, else hidden
            sel = is_encoder[:, None]
            val = tl.where(sel, enc_val, hid_val)
            # Store to out
            out_ptrs = out_ptr + b * out_stride_b + (l_idx[:, None] + T) * out_stride_l + jj * out_stride_h  # (l_idx + T) maps to [0, T+I)
            tl.store(out_ptrs, val, mask=mask_l[:, None])


@triton.jit
def _split_streams_kernel(
    C_ptr,              # *ptr to processed [B*(T+I), H]
    enc_ptr,            # *ptr to output encoder [B, T, H]
    hid_ptr,            # *ptr to output hidden [B, I, H]
    B, T, I, H,
    C_stride_m, C_stride_n,
    enc_stride_b, enc_stride_t, enc_stride_h,
    hid_stride_b, hid_stride_i, hid_stride_h,
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)  # one program per batch
    # For each l in [0, T) and [0, I), map to m = b*(T+I) + l and copy C[m, :]
    # We'll loop over H in chunks for simplicity
    for l in range(0, T):
        m = b * (T + I) + l
        for h0 in range(0, H, BLOCK_H):
            h_idx = h0 + tl.arange(0, BLOCK_H)
            mask_h = h_idx < H
            # Load C[m, :]
            c_ptrs = C_ptr + m * C_stride_m + h_idx * C_stride_n
            vals = tl.load(c_ptrs, mask=mask_h, other=0.0)
            # Store to encoder
            enc_ptrs = enc_ptr + b * enc_stride_b + l * enc_stride_t + h_idx * enc_stride_h
            tl.store(enc_ptrs, vals, mask=mask_h)
    for l in range(0, I):
        m = b * (T + I) + (T + l)
        for h0 in range(0, H, BLOCK_H):
            h_idx = h0 + tl.arange(0, BLOCK_H)
            mask_h = h_idx < H
            c_ptrs = C_ptr + m * C_stride_m + h_idx * C_stride_n
            vals = tl.load(c_ptrs, mask=mask_h, other=0.0)
            hid_ptrs = hid_ptr + b * hid_stride_b + l * hid_stride_i + h_idx * hid_stride_h
            tl.store(hid_ptrs, vals, mask=mask_h)

# ModelNew forward: Triton-only data movement; matmul via torch for correctness
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Expect dtype float32, device CUDA
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA device"
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors"

        B, T, H = encoder_hidden_states.shape
        assert hidden_states.shape == (B, T, H), "hidden_states must have shape [B, T, H]"
        I = hidden_states.shape[1]
        assert process_weight.shape == (H, H), "process_weight must have shape [H, H]"

        # 1) Concatenate along sequence dimension: out_cat [B, T+I, H]
        out_cat = torch.empty((B, T + I, H), dtype=torch.float32, device=hidden_states.device)

        _concatenate_sequences_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            *encoder_hidden_states.stride(), *hidden_states.stride(),
            *out_cat.stride(),
            BLOCK_L=256,
            num_warps=1, num_stages=1,
        )

        # 2) Linear projection: processed = out_cat @ process_weight.T
        # Use torch.matmul for robustness; this is the compute-heavy part.
        processed = torch.matmul(out_cat, process_weight.t())

        # 3) Split into encoder and hidden streams
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=processed.device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=processed.device)

        # Flatten processed for input to Triton split: [B*(T+I), H]
        C = processed.reshape(B * (T + I), H).contiguous()

        _split_streams_kernel[(B,)](
            C, processed_encoder, processed_hidden,
            B, T, I, H,
            *C.stride(),
            *processed_encoder.stride(), *processed_hidden.stride(),
            BLOCK_H=256,
            num_warps=1, num_stages=1,
        )

        return processed_encoder, processed_hidden