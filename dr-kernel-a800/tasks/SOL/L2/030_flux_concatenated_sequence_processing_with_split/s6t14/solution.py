import torch
import triton
import triton.language as tl


@triton.jit
def _concat_encoder_kernel(
    A_ptr,        # *f32, [B, M, H]
    Out_ptr,      # *f32, [B, C, H], C = M + N
    B: tl.constexpr,    # batch size
    M: tl.constexpr,    # text_seq_len
    H: tl.constexpr,    # hidden_dim
    stride_ab,    # int: stride along batch for A (elements)
    stride_am,    # int: stride along seq for A (elements)
    stride_ah,    # int: stride along hidden for A (elements)
    stride_ob,    # int: stride along batch for Out (elements)
    stride_om,    # int: stride along seq for Out (elements)
    stride_oh,    # int: stride along hidden for Out (elements)
    BLOCK_M: tl.constexpr,  # tile over sequence M
    BLOCK_H: tl.constexpr,  # tile over hidden H
):
    # Grid: (B, ceil_div(M, BLOCK_M))
    b = tl.program_id(0)
    m_block = tl.program_id(1)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_m = m_offsets < M

    # Iterate over hidden dimension in chunks
    for h0 in range(0, H, BLOCK_H):
        h_offsets = h0 + tl.arange(0, BLOCK_H)  # [BLOCK_H]
        mask_h = h_offsets < H

        # Load A[b, m, h]
        a_ptrs = A_ptr + b * stride_ab + m_offsets[:, None] * stride_am + h_offsets[None, :] * stride_ah  # [BLOCK_M, BLOCK_H]
        a_mask = mask_m[:, None] & mask_h[None, :]
        a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Store into Out[b, m, h] in first M positions
        out_ptrs = Out_ptr + b * stride_ob + m_offsets[:, None] * stride_om + h_offsets[None, :] * stride_oh
        tl.store(out_ptrs, a_vals, mask=a_mask)


@triton.jit
def _concat_hidden_kernel(
    B_ptr,        # *f32, [B, N, H]
    Out_ptr,      # *f32, [B, C, H], C = M + N
    B: tl.constexpr,    # batch size
    M: tl.constexpr,    # text_seq_len
    N: tl.constexpr,    # img_seq_len
    H: tl.constexpr,    # hidden_dim
    stride_bb,    # int: stride along batch for B (elements)
    stride_bn,    # int: stride along seq for B (elements)
    stride_bh,    # int: stride along hidden for B (elements)
    stride_ob,    # int: stride along batch for Out (elements)
    stride_om,    # int: stride along seq for Out (elements)
    stride_oh,    # int: stride along hidden for Out (elements)
    BLOCK_N: tl.constexpr,  # tile over sequence N
    BLOCK_H: tl.constexpr,  # tile over hidden H
):
    # Grid: (B, ceil_div(N, BLOCK_N))
    b = tl.program_id(0)
    n_block = tl.program_id(1)

    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_n = n_offsets < N

    # Iterate over hidden dimension in chunks
    for h0 in range(0, H, BLOCK_H):
        h_offsets = h0 + tl.arange(0, BLOCK_H)  # [BLOCK_H]
        mask_h = h_offsets < H

        # Load B[b, n, h]
        b_ptrs = B_ptr + b * stride_bb + n_offsets[:, None] * stride_bn + h_offsets[None, :] * stride_bh  # [BLOCK_N, BLOCK_H]
        b_mask = mask_n[:, None] & mask_h[None, :]
        b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Store into Out[b, M + n, h]
        out_ptrs = Out_ptr + b * stride_ob + (n_offsets[:, None] + M) * stride_om + h_offsets[None, :] * stride_oh
        tl.store(out_ptrs, b_vals, mask=b_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
        - Uses Triton kernels to concatenate the sequences along the sequence dimension.
        - Performs the linear projection using PyTorch matmul to ensure numerical correctness.
        Returns (processed_encoder, processed_hidden) as in the original.
        """
        # Ensure CUDA tensors
        if not hidden_states.is_cuda or not encoder_hidden_states.is_cuda or not process_weight.is_cuda:
            raise RuntimeError("All inputs must be CUDA tensors.")

        # Ensure contiguity
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        if not encoder_hidden_states.is_contiguous():
            encoder_hidden_states = encoder_hidden_states.contiguous()
        if not process_weight.is_contiguous():
            process_weight = process_weight.contiguous()

        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        H = hidden_states.shape[2]
        C = M + N

        # Allocate concatenated tensor
        Out = torch.empty((B, C, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernels for concatenation
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_H = 64

        grid_encoder = (B, triton.cdiv(M, BLOCK_M))
        _concat_encoder_kernel[grid_encoder](
            encoder_hidden_states, Out,
            B=B, M=M, H=H,
            stride_ab=encoder_hidden_states.stride(0),
            stride_am=encoder_hidden_states.stride(1),
            stride_ah=encoder_hidden_states.stride(2),
            stride_ob=Out.stride(0),
            stride_om=Out.stride(1),
            stride_oh=Out.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2
        )

        grid_hidden = (B, triton.cdiv(N, BLOCK_N))
        _concat_hidden_kernel[grid_hidden](
            hidden_states, Out,
            B=B, M=M, N=N, H=H,
            stride_bb=hidden_states.stride(0),
            stride_bn=hidden_states.stride(1),
            stride_bh=hidden_states.stride(2),
            stride_ob=Out.stride(0),
            stride_om=Out.stride(1),
            stride_oh=Out.stride(2),
            BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2
        )

        # Linear projection using PyTorch (exact and robust)
        processed = torch.matmul(Out, process_weight.t())  # [B, C, H]

        # Split back into two streams
        processed_encoder = processed[:, :M, :]
        processed_hidden = processed[:, M:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
