import torch
import triton
import triton.language as tl


@triton.jit
def _concat_seq_kernel(
    A_ptr,        # *f32, [B, M, H]
    B_ptr,        # *f32, [B, N, H]
    Out_ptr,      # *f32, [B, C, H], C = M + N
    B: tl.constexpr,    # batch size
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
    BLOCK_M: tl.constexpr,  # tile along sequence dimension
    BLOCK_N: tl.constexpr,  # tile along hidden dimension
):
    # Grid over (batch, tiles along sequence dimension C)
    b = tl.program_id(0)
    tile = tl.program_id(1)

    m_offsets = tile * BLOCK_M + tl.arange(0, BLOCK_M)  # positions from A [0..M)
    n_offsets = tile * BLOCK_N + tl.arange(0, BLOCK_N)  # hidden indices [0..H)

    mask_m = m_offsets < M
    mask_n = n_offsets < N
    mask_h = n_offsets < H

    # Load from A: positions 0..M-1
    a_ptrs = A_ptr + b * stride_ab + m_offsets[:, None] * stride_am + n_offsets[None, :] * stride_ah
    a_mask = mask_m[:, None] & mask_h[None, :]
    a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_N]

    # Load from B: positions M..M+N-1
    b_ptrs = B_ptr + b * stride_bb + (m_offsets - M)[:, None] * stride_bn + n_offsets[None, :] * stride_bh
    b_mask = (~mask_m[:, None]) & mask_h[None, :]
    b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_M, BLOCK_N]

    # Select based on m_offsets < M
    vals = tl.where(m_offsets[:, None] < M, a_vals, b_vals)  # [BLOCK_M, BLOCK_N]

    # Store into Out at positions [b, m_offsets + M, n_offsets]
    out_ptrs = Out_ptr + b * stride_ob + (m_offsets[:, None] + M) * stride_oc + n_offsets[None, :] * stride_oh
    store_mask = (m_offsets[:, None] < (M + N)) & mask_h[None, :]
    tl.store(out_ptrs, vals, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Inputs:
        # hidden_states: [B, N, H]
        # encoder_hidden_states: [B, M, H]
        # process_weight: [H, H]
        # Output: (processed_encoder [B, M, H], processed_hidden [B, N, H])

        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be [B, dim, H]"
        B = hidden_states.shape[0]
        N = hidden_states.shape[1]
        M = encoder_hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "hidden_dim must match for both inputs"
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be CUDA"
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32"

        # Concatenate sequences along dim=1 (sequence length): Out [B, C, H], C = M + N
        C = M + N
        Out = torch.empty((B, C, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel for concatenation
        BLOCK_M = 64
        BLOCK_N = 64
        grid = (B, triton.cdiv(C, BLOCK_M))
        _concat_seq_kernel[grid](
            encoder_hidden_states, hidden_states, Out,
            B=B, M=M, N=N, H=H,
            stride_ab=encoder_hidden_states.stride(0), stride_am=encoder_hidden_states.stride(1), stride_ah=encoder_hidden_states.stride(2),
            stride_bb=hidden_states.stride(0), stride_bn=hidden_states.stride(1), stride_bh=hidden_states.stride(2),
            stride_ob=Out.stride(0), stride_oc=Out.stride(1), stride_oh=Out.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        # Apply linear projection: Out [B, C, H] @ process_weight.T [H, H] -> P [B, C, H]
        # Use torch.matmul to ensure exact numerical match with PyTorch.
        W = process_weight  # [H, H]
        P = torch.matmul(Out, W.t())  # [B, C, H]

        # Split back into two streams
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
