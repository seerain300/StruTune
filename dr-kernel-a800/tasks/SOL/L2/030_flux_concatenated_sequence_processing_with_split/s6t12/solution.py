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
    stride_ab,    # int: stride along batch for A
    stride_am,    # int: stride along seq for A
    stride_ah,    # int: stride along hidden for A
    stride_ob,    # int: stride along batch for Out
    stride_om,    # int: stride along seq for Out
    stride_oh,    # int: stride along hidden for Out
    BLOCK_M: tl.constexpr,  # tile over sequence M
    BLOCK_H: tl.constexpr,  # tile over hidden H
):
    # Grid: (B, ceil_div(M, BLOCK_M))
    b = tl.program_id(0)
    m_block = tl.program_id(1)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_m = m_offsets < M

    for h0 in range(0, H, BLOCK_H):
        h_offsets = h0 + tl.arange(0, BLOCK_H)  # [BLOCK_H]
        mask_h = h_offsets < H

        a_ptrs = A_ptr + b * stride_ab + m_offsets[:, None] * stride_am + h_offsets[None, :] * stride_ah  # [BLOCK_M, BLOCK_H]
        a_mask = mask_m[:, None] & mask_h[None, :]
        a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)

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
    stride_bb,    # int: stride along batch for B
    stride_bn,    # int: stride along seq for B
    stride_bh,    # int: stride along hidden for B
    stride_ob,    # int: stride along batch for Out
    stride_om,    # int: stride along seq for Out
    stride_oh,    # int: stride along hidden for Out
    BLOCK_N: tl.constexpr,  # tile over sequence N
    BLOCK_H: tl.constexpr,  # tile over hidden H
):
    # Grid: (B, ceil_div(N, BLOCK_N))
    b = tl.program_id(0)
    n_block = tl.program_id(1)

    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    seq_offsets = M + n_offsets  # absolute sequence indices in Out
    mask_n = n_offsets < N

    for h0 in range(0, H, BLOCK_H):
        h_offsets = h0 + tl.arange(0, BLOCK_H)  # [BLOCK_H]
        mask_h = h_offsets < H

        b_ptrs = B_ptr + b * stride_bb + n_offsets[:, None] * stride_bn + h_offsets[None, :] * stride_bh  # [BLOCK_N, BLOCK_H]
        b_mask = mask_n[:, None] & mask_h[None, :]
        b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)

        out_ptrs = Out_ptr + b * stride_ob + seq_offsets[:, None] * stride_om + h_offsets[None, :] * stride_oh
        tl.store(out_ptrs, b_vals, mask=b_mask)


@triton.jit
def _batched_gemm_kernel(
    X_ptr,  # *f32, [B, C, K] where C = M + N, K = H
    W_ptr,  # *f32, [K, K] (process_weight)
    P_ptr,  # *f32, [B, C, K]
    B: tl.constexpr,      # batch size
    C: tl.constexpr,      # sequence length (M + N)
    K: tl.constexpr,      # hidden dim
    # Strides for X: [B, C, K]
    stride_xb,  # int
    stride_xc,  # int
    stride_xk,  # int
    # Strides for W: [K, K]
    stride_w0,  # int
    stride_w1,  # int
    # Strides for P: [B, C, K]
    stride_pb,  # int
    stride_pc,  # int
    stride_pk,  # int
    BLOCK_M: tl.constexpr,  # tile over C
    BLOCK_N: tl.constexpr,  # tile over K
    BLOCK_K: tl.constexpr,  # tile over reduction K
):
    # 2D grid: (B, ceil_div(C, BLOCK_M))
    b = tl.program_id(0)
    c_block = tl.program_id(1)

    # tile offsets
    c_offsets = c_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # masks for boundaries
    mask_c = c_offsets < C
    mask_k = k_offsets < K

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        # load X tile: [BLOCK_M, BLOCK_K] = X[b, c, k0:k0+BLOCK_K]
        x_ptrs = X_ptr + b * stride_xb + c_offsets[:, None] * stride_xc + (k0 + tl.arange(0, BLOCK_K))[None, :] * stride_xk
        x_mask = mask_c[:, None] & ((k0 + tl.arange(0, BLOCK_K))[None, :] < K)
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # load W tile as [BLOCK_K, BLOCK_N] where W[k, k'] with k in [k0:..], k' in [k_offsets]
        w_ptrs = W_ptr + (k0 + tl.arange(0, BLOCK_K))[:, None] * stride_w0 + k_offsets[None, :] * stride_w1
        w_mask = ((k0 + tl.arange(0, BLOCK_K))[:, None] < K) & (k_offsets[None, :] < K)
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # accumulate
        acc += tl.dot(x_vals, w_vals)  # [BLOCK_M, BLOCK_N]

    # Store results: P[b, c, k_offsets]
    p_ptrs = P_ptr + b * stride_pb + c_offsets[:, None] * stride_pc + k_offsets[None, :] * stride_pk
    store_mask = mask_c[:, None] & mask_k[None, :]
    tl.store(p_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension using Triton.
        - Apply linear projection using a Triton GEMM: X @ process_weight.T.
        - Split back into (processed_encoder, processed_hidden).
        """
        # Ensure CUDA tensors and dtype float32
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA"
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors"
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B, M, H = encoder_hidden_states.shape
        N = hidden_states.shape[1]
        C = M + N

        # Allocate concatenated tensor Out [B, C, H]
        Out = torch.empty((B, C, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton concatenation kernels
        BLOCK_M = 64
        BLOCK_H = 128
        grid_encoder = (B, triton.cdiv(M, BLOCK_M))
        _concat_encoder_kernel[grid_encoder](
            encoder_hidden_states, Out,
            B=B, M=M, H=H,
            stride_ab=encoder_hidden_states.stride(0), stride_am=encoder_hidden_states.stride(1), stride_ah=encoder_hidden_states.stride(2),
            stride_ob=Out.stride(0), stride_om=Out.stride(1), stride_oh=Out.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2
        )

        BLOCK_N = 64
        grid_hidden = (B, triton.cdiv(N, BLOCK_N))
        _concat_hidden_kernel[grid_hidden](
            hidden_states, Out,
            B=B, M=M, N=N, H=H,
            stride_bb=hidden_states.stride(0), stride_bn=hidden_states.stride(1), stride_bh=hidden_states.stride(2),
            stride_ob=Out.stride(0), stride_om=Out.stride(1), stride_oh=Out.stride(2),
            BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2
        )

        # Allocate output of GEMM
        P = torch.empty((B, C, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton GEMM: P = Out @ process_weight.T
        BLOCK_M_GEMM = 64
        BLOCK_N_GEMM = 64
        BLOCK_K_GEMM = 64
        grid_gemm = (B, triton.cdiv(C, BLOCK_M_GEMM))
        _batched_gemm_kernel[grid_gemm](
            Out, process_weight, P,
            B=B, C=C, K=H,
            stride_xb=Out.stride(0), stride_xc=Out.stride(1), stride_xk=Out.stride(2),
            stride_w0=process_weight.stride(0), stride_w1=process_weight.stride(1),
            stride_pb=P.stride(0), stride_pc=P.stride(1), stride_pk=P.stride(2),
            BLOCK_M=BLOCK_M_GEMM, BLOCK_N=BLOCK_N_GEMM, BLOCK_K=BLOCK_K_GEMM,
            num_warps=4, num_stages=2
        )

        # Split along sequence dimension
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
