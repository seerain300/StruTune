import torch
import triton
import triton.language as tl

@triton.jit
def _concatenate_seqs_kernel(
    enc_ptr,              # *float/half, [B, T, H]
    hid_ptr,              # *float/half, [B, I, H]
    out_ptr,              # *float/half, [B, L, H], L = T + I
    B: tl.constexpr,      # int
    T: tl.constexpr,      # int
    I: tl.constexpr,      # int
    H: tl.constexpr,      # int
    BLOCK_M: tl.constexpr,  # tile along seq (M=L)
    BLOCK_K: tl.constexpr,  # tile along H (K=H)
):
    # Grid: (B, ceil(L/BLOCK_M))
    b = tl.program_id(0)
    m_block = tl.program_id(1)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # along L (T+I)
    k_offsets = tl.arange(0, BLOCK_K)                      # along H

    L = T + I
    for m in range(0, BLOCK_M):
        m_idx = m_offsets[m]
        # if m_idx < L:
        # Determine source tensor
        src_enc = m_idx < T
        # Compute pointers
        enc_ptr_row = enc_ptr + b * enc_ptr.stride(0) + m_idx * enc_ptr.stride(1) + k_offsets * enc_ptr.stride(2)
        hid_ptr_row = hid_ptr + b * hid_ptr.stride(0) + (m_idx - T) * hid_ptr.stride(1) + k_offsets * hid_ptr.stride(2)
        out_ptr_row = out_ptr + b * out_ptr.stride(0) + m_idx * out_ptr.stride(1) + k_offsets * out_ptr.stride(2)

        enc_val = tl.load(enc_ptr_row, mask=src_enc, other=0.0)
        hid_val = tl.load(hid_ptr_row, mask=~src_enc, other=0.0)  # only valid when m_idx >= T

        # Select based on src_enc
        val = tl.where(src_enc, enc_val, hid_val)
        tl.store(out_ptr_row, val)

# Define ModelNew with Triton-only concatenation and PyTorch matmul for projection
class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, H]
        encoder_hidden_states: [B, T, H]
        process_weight: [H, H]
        Returns: (processed_encoder: [B, T, H], processed_hidden: [B, I, H])
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be [B, dim, H]"
        assert process_weight.dim() == 2, "process_weight must be [H, H]"
        B = hidden_states.size(0)
        T = encoder_hidden_states.size(1)
        I = hidden_states.size(1)
        H = hidden_states.size(2)
        assert encoder_hidden_states.size(2) == H and process_weight.size(1) == H and process_weight.size(0) == H, "Mismatched hidden sizes"

        # Make inputs contiguous
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        device = enc.device

        # 1) Concatenate sequences along seq dim using Triton: out_cat [B, L, H]
        L = T + I
        out_cat = torch.empty((B, L, H), device=device, dtype=enc.dtype)

        # Strides
        strideB, strideM, strideK = out_cat.stride()
        strideB_enc, strideM_enc, strideK_enc = enc.stride()
        strideB_hid, strideM_hid, strideK_hid = hid.stride()

        # Launch Triton kernel: grid over batch and tiles along L
        # Choose small blocks to cover general sizes; L,H are dynamic
        BLOCK_M = 128  # tile along L
        BLOCK_K = 64   # tile along H
        grid = (B, triton.cdiv(L, BLOCK_M))

        _concatenate_seqs_kernel[grid](
            enc, hid, out_cat,
            B, T, I, H,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) Linear projection: out_cat @ process_weight.T using PyTorch (robust and correct)
        WT = process_weight.t().contiguous()  # [H, H]
        processed = torch.matmul(out_cat, WT)

        # 3) Split back: processed has shape [B, L, H]
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
