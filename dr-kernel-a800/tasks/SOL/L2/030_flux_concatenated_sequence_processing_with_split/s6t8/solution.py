import torch
import triton
import triton.language as tl


@triton.jit
def _concat_seq_kernel(
    A_ptr,        # *f32, [B, M, H]
    B_ptr,        # *f32, [B, N, H]
    Out_ptr,      # *f32, [B, C, H], C = M + N
    B: tl.constexpr,    # batch size (constexpr for specialization)
    M: tl.constexpr,    # text_seq_len
    N: tl.constexpr,    # img_seq_len
    H: tl.constexpr,    # hidden_dim
    stride_ab,    # int: stride along batch for A
    stride_am,    # int: stride along seq for A
    stride_ah,    # int: stride along hidden for A
    stride_bb,    # int: stride along batch for B
    stride_bn,    # int: stride along seq for B
    stride_bh,    # int: stride along hidden for B
    stride_ob,    # int: stride along batch for Out
    stride_oc,    # int: stride along seq for Out
    stride_oh,    # int: stride along hidden for Out
    BLOCK_M: tl.constexpr,  # tile along seq (M+N)
    BLOCK_H: tl.constexpr,  # tile along hidden
):
    # Grid: (B, ceil_div(C, BLOCK_M), ceil_div(H, BLOCK_H))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    h_block = tl.program_id(2)

    # Sequence offsets for this tile
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M], where C = M + N
    h_offsets = h_block * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]

    # Masks
    mask_m = m_offsets < (M + N)
    mask_h = h_offsets < H

    # Determine source (A or B) for each sequence position
    from_A = m_offsets < M  # boolean per m

    # Compute input pointers and masks
    # A[b, m, h]
    a_ptrs = A_ptr + b * stride_ab + m_offsets[:, None] * stride_am + h_offsets[None, :] * stride_ah
    a_mask = mask_m[:, None] & mask_h[None, :]
    a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)

    # B[b, m-M, h]
    b_ptrs = B_ptr + b * stride_bb + (m_offsets - M)[:, None] * stride_bn + h_offsets[None, :] * stride_bh
    b_mask = (~from_A[:, None]) & mask_m[:, None] & mask_h[None, :]
    b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)

    # Select based on from_A
    vals = tl.where(from_A[:, None], a_vals, b_vals)  # [BLOCK_M, H]

    # Store to Out
    out_ptrs = Out_ptr + b * stride_ob + m_offsets[:, None] * stride_oc + h_offsets[None, :] * stride_oh
    tl.store(out_ptrs, vals, mask=(mask_m[:, None] & mask_h[None, :]))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-based concatenation followed by PyTorch matmul for the projection.
        Returns:
          processed_encoder: [B, M, H]
          processed_hidden:  [B, N, H]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be CUDA."
        assert hidden_states.is_contiguous() and encoder_hidden_states.is_contiguous() and process_weight.is_contiguous(), "All tensors must be contiguous."
        assert hidden_states.shape[0] == encoder_hidden_states.shape[0], "Batch sizes must match."
        assert hidden_states.shape[2] == encoder_hidden_states.shape[2] == process_weight.shape[0] == process_weight.shape[1], "Hidden dims must match and process_weight must be square."

        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        H = hidden_states.shape[2]
        C = M + N

        # Allocate output X [B, C, H]
        X = torch.empty((B, C, H), device=hidden_states.device, dtype=torch.float32)

        # Launch Triton concatenation kernel
        BLOCK_M = 64
        BLOCK_H = 64
        grid = (B, triton.cdiv(C, BLOCK_M), triton.cdiv(H, BLOCK_H))
        _concat_seq_kernel[grid](
            encoder_hidden_states,
            hidden_states,
            X,
            B, M, N, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            X.stride(0), X.stride(1), X.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        # Apply linear projection: X @ process_weight.T
        # process_weight is [H, H], we need X @ W^T -> result [B, C, H]
        # Ensure weight is on same device and dtype
        W = process_weight  # [H, H]
        processed = torch.matmul(X, W.t())  # [B, C, H], no bias

        # Split back into separate streams
        processed_encoder = processed[:, :M, :]
        processed_hidden = processed[:, M:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
