import torch
import triton
import triton.language as tl

@triton.jit
def _concatenate_seq_kernel(
    encoder_ptr,  # *float32 [B, T, H]
    hidden_ptr,   # *float32 [B, I, H]
    out_ptr,      # *float32 [B, T+I, H]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
):
    # One program per batch
    b = tl.program_id(0)
    # Total sequence length
    L = T + I

    # Loop over sequence positions
    # We'll iterate up to L-1 using masks to avoid out-of-bounds.
    for l in range(0, L):
        # Determine source: if l < T, from encoder; else from hidden
        is_encoder = l < T
        # Compute flat offsets
        # Row-major layout assumed contiguous: [B, L, H] -> index = b*L*H + l*H + h
        # We'll load/store with masks.
        # Note: Triton prefers vectorized operations; here we operate per element.
        # We need to write out[b, l, h] for all h in [0, H).
        # We'll do this in chunks of H (one chunk), but Triton can handle scalar loop too.
        # To keep it simple and safe, we iterate h from 0 to H-1 with masks.

        # Prepare h vector
        h = tl.arange(0, H)
        # Source pointers
        if is_encoder:
            src_ptr = encoder_ptr + b * T * H + l * H + h
        else:
            src_idx_i = l - T
            src_ptr = hidden_ptr + b * I * H + src_idx_i * H + h
        # Destination pointer
        dst_ptr = out_ptr + b * L * H + l * H + h

        # Load and store (no mask needed if H is positive; but keep safe)
        # For safety, construct mask for h
        mask_h = h < H
        vals = tl.load(src_ptr, mask=mask_h, other=0.0)
        tl.store(dst_ptr, vals, mask=mask_h)

@triton.jit
def _split_streams_kernel(
    C_ptr,               # *float32 [B, T+I, H] (result of GEMM)
    out_encoder_ptr,     # *float32 [B, T, H]
    out_hidden_ptr,      # *float32 [B, I, H]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
):
    # One program per batch
    b = tl.program_id(0)
    L = T + I

    # For each row index in C (flattened), map to (batch, seq)
    # We'll iterate over rows and compute seq = idx % L, batch = idx // L
    # But since we have B known, we can iterate over all rows M=B*(T+I).
    M = B * L
    # Use a simple loop; Triton allows python-range loops in kernel when constexpr.
    for idx in range(0, M):
        seq = idx % L
        batch = idx // L
        # Write to encoder if seq < T
        if seq < T:
            dst_encoder = out_encoder_ptr + batch * T * H + seq * H
        else:
            dst_hidden = out_hidden_ptr + batch * I * H + (seq - T) * H
        # Load from C: C has shape [B, L, H] contiguous, row idx corresponds to [batch=batch, seq=seq, :]
        src = C_ptr + batch * L * H + seq * H
        h = tl.arange(0, H)
        mask_h = h < H
        vals = tl.load(src, mask=mask_h, other=0.0)
        # Store to destination
        if seq < T:
            tl.store(dst_encoder, vals, mask=mask_h)
        else:
            tl.store(dst_hidden, vals, mask=mask_h)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-based version of the original run:
        - Concatenate encoder_hidden_states and hidden_states along sequence dim using Triton
        - Multiply by process_weight (no bias) using torch.matmul for robustness
        - Split outputs into encoder and hidden streams using Triton
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H and process_weight.shape[0] == H and process_weight.shape[1] == H, "Dimension mismatch."

        # Allocate concatenated tensor [B, T+I, H] in fp32 for numerical stability
        L = T + I
        out_cat = torch.empty((B, L, H), dtype=torch.float32, device=hidden_states.device)

        # Launch concat kernel: one program per batch
        _concatenate_seq_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B=B, T=T, I=I, H=H,
            num_warps=1, num_stages=1
        )

        # Ensure process_weight is fp32 and on device
        W = process_weight
        if W.dtype != torch.float32:
            W = W.float()
        if W.device != hidden_states.device:
            W = W.to(hidden_states.device)

        # Perform GEMM: C = out_cat @ W^T  -> [B, T+I, H]
        # Use torch for robust and fast GEMM
        C = torch.matmul(out_cat, W.t())

        # Allocate outputs [B, T, H] and [B, I, H]
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=hidden_states.device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=hidden_states.device)

        # Launch split kernel: one program per batch
        _split_streams_kernel[(B,)](
            C, processed_encoder, processed_hidden,
            B=B, T=T, I=I, H=H,
            num_warps=1, num_stages=1
        )

        return processed_encoder, processed_hidden