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
    BLOCK_M: tl.constexpr,  # tile along sequence (C)
    BLOCK_H: tl.constexpr,  # tile along hidden dim (H)
):
    # Grid: (B, ceil_div(C, BLOCK_M))
    b = tl.program_id(0)
    c_block = tl.program_id(1)
    C = M + N

    # sequence offsets this program handles
    m_offsets = c_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_c = m_offsets < C

    # hidden dim offsets
    h_offsets = tl.arange(0, H)  # [H]

    # Determine which m belong to A vs B
    from_A = m_offsets < M
    m_in_B = m_offsets - M  # valid where from_A is False

    # Compute pointers for A and B sources
    # A[b, m, h]
    a_ptrs = A_ptr + b * stride_ab + m_offsets[:, None] * stride_am + h_offsets[None, :] * stride_ah
    a_mask = mask_c[:, None]
    a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, H]

    # B[b, m-N, h] for m >= M
    b_ptrs = B_ptr + b * stride_bb + m_in_B[:, None] * stride_bn + h_offsets[None, :] * stride_bh
    b_mask = (~from_A)[:, None] & mask_c[:, None]
    b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_M, H]

    # Select values based on from_A
    vals = tl.where(from_A[:, None], a_vals, b_vals)  # [BLOCK_M, H]

    # Store to Out[b, m_offsets, h_offsets]
    out_ptrs = Out_ptr + b * stride_ob + m_offsets[:, None] * stride_oc + h_offsets[None, :] * stride_oh
    store_mask = mask_c[:, None]
    tl.store(out_ptrs, vals, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, img_seq_len, H] - image latent sequence
        encoder_hidden_states: [B, text_seq_len, H] - text conditioning sequence
        process_weight: [H, H] - linear projection weight
        Returns: (processed_encoder, processed_hidden)
        """
        # Ensure CUDA and dtype
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32."

        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]  # text_seq_len
        N = hidden_states.shape[1]          # img_seq_len
        H = hidden_states.shape[2]          # hidden_dim
        assert process_weight.shape[0] == H and process_weight.shape[1] == H

        # Allocate output for concatenation
        C = M + N
        X = torch.empty((B, C, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch Triton kernel to concatenate along sequence dim
        # Grid: (B, ceil_div(C, BLOCK_M))
        BLOCK_M = 128
        BLOCK_H = 128
        grid = (B, triton.cdiv(C, BLOCK_M))
        _concat_seq_kernel[grid](
            encoder_hidden_states, hidden_states, X,
            B=B, M=M, N=N, H=H,
            stride_ab=encoder_hidden_states.stride(0), stride_am=encoder_hidden_states.stride(1), stride_ah=encoder_hidden_states.stride(2),
            stride_bb=hidden_states.stride(0), stride_bn=hidden_states.stride(1), stride_bh=hidden_states.stride(2),
            stride_ob=X.stride(0), stride_oc=X.stride(1), stride_oh=X.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
        )

        # Compute linear projection using torch (robust and correct)
        # X: [B, C, H], process_weight: [H, H]
        # Result: [B, C, H]
        P = torch.matmul(X, process_weight.transpose(0, 1))

        # Split back into two streams
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
