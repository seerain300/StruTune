import torch
import triton
import triton.language as tl

# Triton kernel: concatenate along sequence dimension for each batch
@triton.jit
def _concatenate_sequences_kernel(
    enc_ptr,        # *f32, shape [B, T, H]
    hid_ptr,        # *f32, shape [B, I, H]
    out_ptr,        # *f32, shape [B, T+I, H]
    B, T, I, H,     # int32
):
    b = tl.program_id(0)  # one program per batch
    if b >= 0:
        # Loop over T rows: write to out[b, 0:T, :]
        for l in range(0, T):
            offs = tl.arange(0, H)
            enc_row = enc_ptr + b * T * H + l * H + offs
            out_row = out_ptr + b * (T + I) * H + l * H + offs
            # mask for H-dimension
            tl.store(out_row, tl.load(enc_row), mask=offs < H)

        # Loop over I rows: write to out[b, T:T+I, :]
        for l in range(0, I):
            offs = tl.arange(0, H)
            hid_row = hid_ptr + b * I * H + l * H + offs
            out_row = out_ptr + b * (T + I) * H + (T + l) * H + offs
            tl.store(out_row, tl.load(hid_row), mask=offs < H)

# Triton kernel: split along sequence dimension into two outputs for each batch
@triton.jit
def _split_streams_kernel(
    in_ptr,         # *f32, shape [B*(T+I), H], but we will pass the actual C
    out_enc_ptr,    # *f32, shape [B, T, H]
    out_hid_ptr,    # *f32, shape [B, I, H]
    B, T, I, H,     # int32
):
    b = tl.program_id(0)
    if b >= 0:
        # Process encoder part: rows [0, T)
        for l in range(0, T):
            offs = tl.arange(0, H)
            src_row = in_ptr + (b * (T + I) + l) * H + offs
            dst_row = out_enc_ptr + b * T * H + l * H + offs
            tl.store(dst_row, tl.load(src_row), mask=offs < H)

        # Process hidden part: rows [T, T+I)
        for l in range(0, I):
            offs = tl.arange(0, H)
            src_row = in_ptr + (b * (T + I) + T + l) * H + offs
            dst_row = out_hid_ptr + b * I * H + l * H + offs
            tl.store(dst_row, tl.load(src_row), mask=offs < H)

class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,        # [B, I, H]
        encoder_hidden_states: torch.Tensor, # [B, T, H]
        process_weight: torch.Tensor,       # [H, H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenate encoder_hidden_states and hidden_states along sequence dim (Triton).
        - Apply linear projection via torch.matmul (robust and fast).
        - Split back into two streams (Triton).
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D tensors [B, L, H]"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H and process_weight.shape[0] == H and process_weight.shape[1] == H, "Hidden dims must match"

        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1) Concatenate sequences along sequence dimension using Triton
        out_cat = torch.empty((B, T + I, H), dtype=dtype, device=device)
        _concatenate_sequences_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            num_warps=1, num_stages=1
        )

        # 2) Linear projection: out_cat @ process_weight.T
        # torch.matmul is efficient and ensures correctness across diverse shapes.
        # Note: process_weight is [H, H]; we need W^T for right-multiplication.
        Wt = process_weight.t().contiguous()  # [H, H]
        # out_cat shape: [B, T+I, H]
        # result: [B, T+I, H]
        processed = torch.matmul(out_cat, Wt)

        # 3) Split into encoder and hidden outputs using Triton
        processed_encoder = torch.empty((B, T, H), dtype=dtype, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=dtype, device=device)
        _split_streams_kernel[(B,)](
            processed, processed_encoder, processed_hidden,
            B, T, I, H,
            num_warps=1, num_stages=1
        )

        return processed_encoder, processed_hidden