import torch
import triton
import triton.language as tl


@triton.jit
def _concat_seq_kernel(
    A_ptr,        # *f32, [B, M, H]
    B_ptr,        # *f32, [B, N, H]
    Out_ptr,      # *f32, [B, C, H], C = M + N
    B: tl.constexpr,    # batch size (constexpr)
    M: tl.constexpr,    # text_seq_len (constexpr)
    N: tl.constexpr,    # img_seq_len (constexpr)
    H: tl.constexpr,    # hidden_dim (constexpr)
    stride_ab,    # int: stride along batch for A
    stride_am,    # int: stride along seq for A
    stride_ah,    # int: stride along hidden for A
    stride_bb,    # int: stride along batch for B
    stride_bn,    # int: stride along seq for B
    stride_bh,    # int: stride along hidden for B
    stride_ob,    # int: stride along batch for Out
    stride_oc,    # int: stride along seq for Out
    stride_oh,    # int: stride along hidden for Out
    BLOCK_M: tl.constexpr,  # tile along M (e.g., 64)
    BLOCK_N: tl.constexpr,  # tile along N (e.g., 64)
):
    # Each program handles a block of M and N for one batch b
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    # Compute sequence indices this program will handle
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Masks for boundaries
    mask_m = m_offsets < M
    mask_n = n_offsets < N

    # For each hidden dimension h in [0, H)
    for h in range(0, H):
        # Load from A: A[b, m, h]
        a_ptrs = A_ptr + b * stride_ab + m_offsets * stride_am + h * stride_ah
        a_vals = tl.load(a_ptrs, mask=mask_m, other=0.0)  # [BLOCK_M]

        # Load from B: B[b, n, h]
        b_ptrs = B_ptr + b * stride_bb + n_offsets * stride_bn + h * stride_bh
        b_vals = tl.load(b_ptrs, mask=mask_n, other=0.0)  # [BLOCK_N]

        # Store into Out at two locations:
        # Out[b, m, h] for m in [0, M)
        out_a_ptrs = Out_ptr + b * stride_ob + m_offsets * stride_oc + h * stride_oh
        tl.store(out_a_ptrs, a_vals, mask=mask_m)

        # Out[b, M + n, h] for n in [0, N)
        out_b_ptrs = Out_ptr + b * stride_ob + (m_offsets + M) * stride_oc + h * stride_oh
        tl.store(out_b_ptrs, b_vals, mask=mask_n)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-orchestrated version:
          - Concatenates encoder_hidden_states and hidden_states along sequence dimension in Triton.
          - Performs the linear projection using torch.matmul (ensures exact numerical match).
          - Splits the result back into separate streams.

        Args:
            hidden_states: Image latent sequence [batch, img_seq_len, hidden_dim]
            encoder_hidden_states: Text conditioning sequence [batch, text_seq_len, hidden_dim]
            process_weight: Linear projection weight [hidden_dim, hidden_dim]
        Returns:
            Tuple of (processed_encoder_hidden_states, processed_hidden_states)
        """
        # Ensure CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors."

        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        H = hidden_states.shape[2]

        A = encoder_hidden_states.contiguous()    # [B, M, H]
        B_b = hidden_states.contiguous()          # [B, N, H]
        W = process_weight.contiguous()           # [H, H]

        # Allocate output for concatenation
        C = M + N
        Out = torch.empty((B, C, H), device=hidden_states.device, dtype=torch.float32)

        # Launch Triton concat kernel: grid over (B, tiles along M and N)
        grid = (B, triton.cdiv(M, 64), triton.cdiv(N, 64))
        _concat_seq_kernel[grid](
            A, B_b, Out,
            B, M, N, H,
            A.stride(0), A.stride(1), A.stride(2),
            B_b.stride(0), B_b.stride(1), B_b.stride(2),
            Out.stride(0), Out.stride(1), Out.stride(2),
            BLOCK_M=64, BLOCK_N=64,
        )

        # Apply linear projection using torch.matmul (no bias). Out: [B, C, H], W: [H, H] -> P: [B, C, H]
        P = torch.matmul(Out, W.t())

        # Split back along sequence dimension
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
