import torch
import triton
import triton.language as tl


@triton.jit
def _concat_seqs_kernel(
    A_ptr,        # *f32, [B, M, H] (encoder)
    B_ptr,        # *f32, [B, N, H] (image)
    Out_ptr,      # *f32, [B, C, H], C = M + N
    B: tl.constexpr,     # batch size
    M: tl.constexpr,     # text_seq_len
    N: tl.constexpr,     # img_seq_len
    H: tl.constexpr,     # hidden_dim
    stride_ab,    # int: stride for batch in A
    stride_am,    # int: stride for seq in A
    stride_ah,    # int: stride for hidden in A
    stride_bb,    # int: stride for batch in B
    stride_bn,    # int: stride for seq in B
    stride_bh,    # int: stride for hidden in B
    stride_ob,    # int: stride for batch in Out
    stride_oc,    # int: stride for seq in Out
    stride_oh,    # int: stride for hidden in Out
    BLOCK: tl.constexpr, # tile size along sequence
):
    # program ids: batch and tile along sequence
    b = tl.program_id(0)
    tile_id = tl.program_id(1)

    # offsets along sequence for this tile
    m_offsets = tile_id * BLOCK + tl.arange(0, BLOCK)  # [BLOCK]

    # overall in-range mask for output
    in_range = m_offsets < (M + N)

    # Determine which offsets belong to encoder (first M) and which to image (next N)
    from_encoder = m_offsets < M
    from_image = ~from_encoder

    # Compute base pointers for Out rows
    out_row_base = Out_ptr + b * stride_ob + m_offsets * stride_oc  # [BLOCK]
    # Hidden dimension offsets for store
    h = tl.arange(0, H) * stride_oh  # [H]
    out_ptrs = out_row_base[:, None] + h[None, :]  # [BLOCK, H]

    # Compute A and B pointers and masks
    a_ptrs = A_ptr + b * stride_ab + m_offsets * stride_am + tl.arange(0, H) * stride_ah  # [BLOCK, H]
    b_ptrs = B_ptr + b * stride_bb + (m_offsets - M) * stride_bn + tl.arange(0, H) * stride_bh  # [BLOCK, H]

    # Masks for loads
    a_mask = (m_offsets < M)[:, None] & (tl.arange(0, H)[None, :] < H)
    b_mask = (m_offsets >= M)[:, None] & (tl.arange(0, H)[None, :] < H)

    # Load values
    a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK, H]
    b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK, H]

    # Select based on from_encoder/from_image
    vals = tl.where(from_encoder[:, None], a_vals, b_vals)  # [BLOCK, H]

    # Store to Out with overall in_range mask
    tl.store(out_ptrs, vals, mask=in_range[:, None])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenate encoder_hidden_states and hidden_states along sequence using Triton,
        then apply linear projection with PyTorch matmul. Return split results.
        """
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors."

        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        H = encoder_hidden_states.shape[2]
        assert hidden_states.shape[2] == H, "hidden_states and encoder_hidden_states must have the same hidden_dim."
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [hidden_dim, hidden_dim]."

        # Make inputs contiguous
        A = encoder_hidden_states.contiguous()  # [B, M, H]
        X = hidden_states.contiguous()          # [B, N, H]
        W = process_weight.contiguous()         # [H, H]

        # Allocate output for concatenation [B, C, H], C = M + N
        C = M + N
        Out = torch.empty((B, C, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton concatenation kernel
        BLOCK = 128  # tile size along sequence; masks handle tails
        grid = (B, triton.cdiv(C, BLOCK))
        _concat_seqs_kernel[grid](
            A, X, Out,
            B=B, M=M, N=N, H=H,
            stride_ab=A.stride(0), stride_am=A.stride(1), stride_ah=A.stride(2),
            stride_bb=X.stride(0), stride_bn=X.stride(1), stride_bh=X.stride(2),
            stride_ob=Out.stride(0), stride_oc=Out.stride(1), stride_oh=Out.stride(2),
            BLOCK=BLOCK,
            num_warps=4, num_stages=2,
        )

        # Perform linear projection with PyTorch matmul (exact match)
        # P = Out @ W^T -> shape [B, C, H]
        P = torch.matmul(Out, W.t())

        # Split back along sequence dimension
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
