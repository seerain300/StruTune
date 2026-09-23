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

    for h0 in range(0, H, BLOCK_H):
        h_offsets = h0 + tl.arange(0, BLOCK_H)  # [BLOCK_H]
        mask_h = h_offsets < H

        # Load from A[b, m, h]
        a_ptrs = A_ptr + b * stride_ab + m_offsets[:, None] * stride_am + h_offsets[None, :] * stride_ah
        a_mask = mask_m[:, None] & mask_h[None, :]
        a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_H]

        # Store into Out[b, m, h] at positions [m, h] in the first M rows
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

    for h0 in range(0, H, BLOCK_H):
        h_offsets = h0 + tl.arange(0, BLOCK_H)  # [BLOCK_H]
        mask_h = h_offsets < H

        # Load from B[b, n, h]
        b_ptrs = B_ptr + b * stride_bb + n_offsets[:, None] * stride_bn + h_offsets[None, :] * stride_bh
        b_mask = mask_n[:, None] & mask_h[None, :]
        b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_N, BLOCK_H]

        # Store into Out[b, M + n, h]
        out_ptrs = Out_ptr + b * stride_ob + (n_offsets[:, None] + M) * stride_om + h_offsets[None, :] * stride_oh
        tl.store(out_ptrs, b_vals, mask=b_mask)


@triton.jit
def _batched_gemm_kernel(
    X_ptr,  # *f32, [B, C, K] (row-major: (b, c, h))
    W_ptr,  # *f32, [K, K] (process_weight)
    P_ptr,  # *f32, [B, C, K]
    B: tl.constexpr,   # batch size
    C: tl.constexpr,   # sequence length (M + N)
    K: tl.constexpr,   # hidden dim
    stride_xb,  # int: stride along batch for X
    stride_xc,  # int: stride along seq for X
    stride_xk,  # int: stride along hidden for X
    stride_w0,  # int: stride for rows in W (dim 0)
    stride_w1,  # int: stride for cols in W (dim 1)
    stride_pb,  # int: stride along batch for P
    stride_pc,  # int: stride along seq for P
    stride_pk,  # int: stride along hidden for P
    BLOCK_M: tl.constexpr,  # tile over C (seq)
    BLOCK_N: tl.constexpr,  # tile over K (hidden)
    BLOCK_K: tl.constexpr,  # tile over reduction dim K
):
    # Grid: (B, ceil_div(C, BLOCK_M), ceil_div(K, BLOCK_N))
    b = tl.program_id(0)
    c_block = tl.program_id(1)
    k_block = tl.program_id(2)

    m_offsets = c_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M], indexes rows in [0, C)
    n_offsets = k_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N], indexes hidden dim

    mask_m = m_offsets < C
    mask_n = n_offsets < K

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_offsets < K

        # Load X tile: [BLOCK_M x BLOCK_K]
        # X[b, m, k] with m=m_offsets, k=k_offsets
        x_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xc + k_offsets[None, :] * stride_xk
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load W tile as [BLOCK_K x BLOCK_N] for dot: we want W^T so [k, n]
        # W[k, n] accessed via W_ptr + k * stride_w0 + n * stride_w1
        w_ptrs = W_ptr + k_offsets[:, None] * stride_w0 + n_offsets[None, :] * stride_w1
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(x_vals, w_vals)  # [BLOCK_M, BLOCK_N]

    # Store results into P[b, m, n]
    p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    p_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(p_ptrs, acc, mask=p_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
        - Uses Triton kernels to concatenate the sequences along the sequence dimension.
        - Performs the linear projection using a Triton GEMM kernel (batched matmul).
        Returns (processed_encoder, processed_hidden) as in the original.
        """
        # Ensure CUDA tensors
        if not hidden_states.is_cuda or not encoder_hidden_states.is_cuda or not process_weight.is_cuda:
            raise RuntimeError("All inputs must be CUDA tensors.")

        # Ensure contiguity and float32 dtype
        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        H = hidden_states.shape[2]

        C = M + N

        # Allocate concatenated tensor
        Out = torch.empty((B, C, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernels for concatenation
        # Tile sizes: moderate and robust
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

        # Prepare process_weight as [H, H] and ensure contiguity
        W = process_weight.contiguous()  # [H, H]

        # Allocate output P = Out @ W^T -> [B, C, H]
        P = torch.empty((B, C, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton GEMM kernel
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64

        grid = (B, triton.cdiv(C, BLOCK_M), triton.cdiv(H, BLOCK_N))
        _batched_gemm_kernel[grid](
            Out, W, P,
            B=B, C=C, K=H,
            stride_xb=Out.stride(0), stride_xc=Out.stride(1), stride_xk=Out.stride(2),
            stride_w0=W.stride(0), stride_w1=W.stride(1),
            stride_pb=P.stride(0), stride_pc=P.stride(1), stride_pk=P.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3
        )

        # Split into encoder and hidden streams
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]
        return processed_encoder, processed_hidden


# Optional: local test (won't run in the evaluation environment unless CUDA/Triton available)
if __name__ == "__main__":
    device = "cuda"
    B, M, N, H = 2, 128, 256, 64
    hidden_states = torch.randn(B, N, H, device=device, dtype=torch.float32)
    encoder_hidden_states = torch.randn(B, M, H, device=device, dtype=torch.float32)
    process_weight = torch.randn(H, H, device=device, dtype=torch.float32)

    model = ModelNew().to(device)
    enc_out, hid_out = model(hidden_states, encoder_hidden_states, process_weight)
    print("Encoder out shape:", enc_out.shape)  # [B, M, H]
    print("Hidden out shape:", hid_out.shape)   # [B, N, H]


def run(*args):
    return ModelNew()(*args)
