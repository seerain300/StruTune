import torch
import triton
import triton.language as tl


@triton.jit
def _concat_seq_kernel(
    A_ptr,        # *f32, [B, M, H]
    B_ptr,        # *f32, [B, N, H]
    Out_ptr,      # *f32, [B, C, H], C = M + N
    B: tl.constexpr,    # batch size (constexpr for specialization)
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
    BLOCK_M: tl.constexpr,  # tile along M (sequences)
    BLOCK_N: tl.constexpr,  # tile along N (sequences)
):
    # Each program handles one batch element
    b = tl.program_id(0)
    # Iterate over M and N with 1D tiles
    for m_start in range(0, M, BLOCK_M):
        m_offsets = m_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
        mask_m = m_offsets < M
        # Load A tile: A[b, m, h]
        h = tl.arange(0, H)  # since H is constexpr, this is fine
        a_ptrs = A_ptr + b * stride_ab + m_offsets[:, None] * stride_am + h[None, :] * stride_ah
        a_mask = mask_m[:, None] & (h[None, :] < H)
        a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, H]
        # Store to Out[b, m, h]
        out_ptrs = Out_ptr + b * stride_ob + m_offsets[:, None] * stride_oc + h[None, :] * stride_oh
        tl.store(out_ptrs, a_vals, mask=mask_m[:, None])

    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_n = n_offsets < N
        # Load B tile: B[b, n, h]
        h = tl.arange(0, H)
        b_ptrs = B_ptr + b * stride_bb + n_offsets[:, None] * stride_bn + h[None, :] * stride_bh
        b_mask = mask_n[:, None] & (h[None, :] < H)
        b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_N, H]
        # Store to Out[b, m=M+n, h] -> Out[b, M + n_start + offset, h]
        out_m = M + n_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        out_ptrs = Out_ptr + b * stride_ob + out_m[:, None] * stride_oc + h[None, :] * stride_oh
        tl.store(out_ptrs, b_vals, mask=mask_n[:, None])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA."
        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        H = hidden_states.shape[2]

        # Make inputs contiguous and float32
        A = encoder_hidden_states.contiguous().to(torch.float32)    # [B, M, H]
        Bseq = hidden_states.contiguous().to(torch.float32)         # [B, N, H]
        W = process_weight.contiguous().to(torch.float32)           # [H, H]

        # Allocate concatenated X [B, C, H]
        C = M + N
        X = torch.empty((B, C, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton concatenation kernel
        BLOCK_M = 128
        BLOCK_N = 128
        grid = (B,)
        _concat_seq_kernel[grid](
            A, Bseq, X,
            B, M, N, H,
            A.stride(0), A.stride(1), A.stride(2),
            Bseq.stride(0), Bseq.stride(1), Bseq.stride(2),
            X.stride(0), X.stride(1), X.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        # Compute P = X @ W^T using PyTorch to ensure exact numerical match
        # X: [B, C, H], W: [H, H] => W.t(): [H, H]
        P = torch.matmul(X, W.t())  # [B, C, H]

        # Split along sequence dimension
        processed_encoder = P[:, :M, :]   # [B, M, H]
        processed_hidden = P[:, M:, :]    # [B, N, H]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
