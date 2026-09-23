import torch
import triton
import triton.language as tl


@triton.jit
def _concat_seq_kernel(
    A_ptr,        # *f32, [B, M, H]
    B_ptr,        # *f32, [B, N, H]
    Out_ptr,      # *f32, [B, C, H], C = M + N
    B: tl.constexpr,   # batch size
    M: tl.constexpr,   # text_seq_len
    N: tl.constexpr,   # img_seq_len
    H: tl.constexpr,   # hidden_dim
    stride_ab,    # int: stride along batch for A
    stride_am,    # int: stride along seq for A
    stride_ah,    # int: stride along hidden for A
    stride_bb,    # int: stride along batch for B
    stride_bn,    # int: stride along seq for B
    stride_bh,    # int: stride along hidden for B
    stride_ob,    # int: stride along batch for Out
    stride_oc,    # int: stride along seq for Out
    stride_oh,    # int: stride along hidden for Out
):
    # Each program handles one batch b and writes one sequence position m in [0, C)
    b = tl.program_id(0)
    m = tl.program_id(1)
    C = M + N
    if m >= C:
        return

    # Determine source tensor: from A if m < M, else from B (offset by M)
    from_A = m < M

    # Compute pointers for loading
    # For A: Out[b, m, h] = A[b, m, h]
    a_ptrs = A_ptr + b * stride_ab + m * stride_am + tl.arange(0, H) * stride_ah
    # For B: Out[b, m, h] = B[b, m - M, h]
    b_ptrs = B_ptr + b * stride_bb + (m - M) * stride_bn + tl.arange(0, H) * stride_bh

    # Load with mask
    if from_A:
        vals = tl.load(a_ptrs, mask=tl.arange(0, H) < H, other=0.0)
    else:
        vals = tl.load(b_ptrs, mask=tl.arange(0, H) < H, other=0.0)

    # Store into Out[b, m, :]
    out_ptrs = Out_ptr + b * stride_ob + m * stride_oc + tl.arange(0, H) * stride_oh
    tl.store(out_ptrs, vals, mask=tl.arange(0, H) < H)


@triton.jit
def _batch_gemm_kernel_simple(
    X_ptr,   # *f32, [B, C, K] (Out)
    W_ptr,   # *f32, [K, K]
    P_ptr,   # *f32, [B, C, K]
    B: tl.constexpr,   # batch size
    C: tl.constexpr,   # sequence length (M + N)
    K: tl.constexpr,   # hidden dim
    stride_xb,  # int: stride along batch for X
    stride_xc,  # int: stride along seq for X
    stride_xk,  # int: stride along hidden for X
    stride_w0,  # int: stride along rows for W (dim 0)
    stride_w1,  # int: stride along cols for W (dim 1)
    stride_pb,  # int: stride along batch for P
    stride_pc,  # int: stride along seq for P
    stride_pk,  # int: stride along hidden for P
):
    # Grid: (B, 1, 1) => each program computes full P[b, :, :]
    b = tl.program_id(0)

    # Initialize accumulator per (m, n) position. We'll compute m and n in nested loops.
    # Note: Triton does not support non-constexpr for-loops over tensors; we implement nested loops over m and n.
    # The loops are over compile-time ranges since C and K are tl.constexpr.
    for m in range(0, C):
        acc = tl.zeros((1,), dtype=tl.float32)  # scalar accumulator for a single row
        # Compute P[b, m, :] as dot(X[b, m, :], W[:, :]) elementwise for each column n in [0, K)
        for n in range(0, K):
            # acc = sum_k X[b, m, k] * W[k, n]
            acc_n = 0.0
            for k in range(0, K):
                x_val = tl.load(X_ptr + b * stride_xb + m * stride_xc + k * stride_xk)
                w_val = tl.load(W_ptr + k * stride_w0 + n * stride_w1)
                acc_n += float(x_val) * float(w_val)
            # Store P[b, m, n] = acc_n
            tl.store(P_ptr + b * stride_pb + m * stride_pc + n * stride_pk, acc_n)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
        - Builds concatenated X [B, M+N, H] via Triton.
        - Computes P = X @ process_weight.T via Triton.
        - Splits P into encoder and image streams.

        Returns:
        (processed_encoder_hidden_states: [B, M, H], processed_hidden_states: [B, N, H])
        """
        # Ensure CUDA and float32 for predictable behavior and correctness
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be CUDA"
        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        H = hidden_states.shape[2]
        C = M + N

        # Cast to float32 to match typical PyTorch defaults and avoid dtype discrepancies
        A = encoder_hidden_states.contiguous().to(torch.float32)
        Bn = hidden_states.contiguous().to(torch.float32)
        W = process_weight.contiguous().to(torch.float32)

        # Allocate output X [B, C, H] and P [B, C, H]
        Out = torch.empty((B, C, H), device=hidden_states.device, dtype=torch.float32)
        P = torch.empty((B, C, H), device=hidden_states.device, dtype=torch.float32)

        # Launch concatenation kernel: grid over (batch, sequence positions)
        grid_concat = (B, C)
        _concat_seq_kernel[grid_concat](
            A, Bn, Out,
            B=B, M=M, N=N, H=H,
            stride_ab=A.stride(0), stride_am=A.stride(1), stride_ah=A.stride(2),
            stride_bb=Bn.stride(0), stride_bn=Bn.stride(1), stride_bh=Bn.stride(2),
            stride_ob=Out.stride(0), stride_oc=Out.stride(1), stride_oh=Out.stride(2),
            num_warps=1, num_stages=1,
        )

        # Launch simple GEMM kernel: grid over (batch,)
        grid_gemm = (B,)
        _batch_gemm_kernel_simple[grid_gemm](
            Out, W, P,
            B=B, C=C, K=H,
            stride_xb=Out.stride(0), stride_xc=Out.stride(1), stride_xk=Out.stride(2),
            stride_w0=W.stride(0), stride_w1=W.stride(1),
            stride_pb=P.stride(0), stride_pc=P.stride(1), stride_pk=P.stride(2),
            num_warps=1, num_stages=1,
        )

        # Split along sequence dimension
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
