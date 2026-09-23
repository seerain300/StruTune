import triton
import triton.language as tl


# Per-token GEMV kernel: one program per token (row), computes y = A_vec @ B_mat
@triton.jit
def gemv_bf16_fp32_kernel(
    A_vec_ptr,  # *bf16, shape [M_tokens]
    B_mat_ptr,  # *bf16, shape [K, N]
    Out_ptr,    # *bf16, shape [M_tokens, N]
    M_tokens: tl.constexpr,  # number of tokens (rows)
    K: tl.constexpr,         # size of A vector
    N: tl.constexpr,         # size of B matrix columns
    stride_A,                # stride for A (elements)
    stride_BM, stride_BN,    # strides for B (rows, cols)
    stride_OutM, stride_OutN, # strides for Out
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= M_tokens:
        return

    out_vec = tl.zeros((N,), dtype=tl.float32)

    k0 = 0
    while k0 < K:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        # Load A_vec chunk (bf16) -> fp32
        a = tl.load(A_vec_ptr + pid * stride_A + k_offsets, mask=k_offsets < K, other=0.0).to(tl.float32)
        # Load B_mat chunk (bf16) -> fp32; shape [BLOCK_K, N]
        b = tl.load(
            B_mat_ptr + k_offsets[:, None] * stride_BM + tl.arange(0, N)[None, :] * stride_BN,
            mask=(k_offsets[:, None] < K) & (tl.arange(0, N)[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        # Accumulate dot over K-chunk
        out_vec += tl.sum(b * a[:, None], axis=0)
        k0 += BLOCK_K

    # Store result to Out as bf16
    n_offsets = tl.arange(0, N)
    tl.store(Out_ptr + pid * stride_OutM + n_offsets * stride_OutN, out_vec.to(tl.bfloat16), mask=n_offsets < N)


# 2D matmul kernel: C = A @ B, fp32 accumulate, bf16 output
@triton.jit
def matmul_bf16_fp32_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_AM, stride_AK,      # strides for A
    stride_BK, stride_BN,      # strides for B
    stride_CM, stride_CN,      # strides for C
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k0 = 0
    while k0 < K:
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        a = tl.load(
            A_ptr + m_offsets[:, None] * stride_AM + k_offsets[None, :] * stride_AK,
            mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        # Load B tile: [BLOCK_K, BLOCK_N]
        b = tl.load(
            B_ptr + k_offsets[:, None] * stride_BK + n_offsets[None, :] * stride_BN,
            mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N),
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(a, b)

        k0 += BLOCK_K

    # Store C tile: bf16 output
    tl.store(
        C_ptr + m_offsets[:, None] * stride_CM + n_offsets[None, :] * stride_CN,
        acc.to(tl.bfloat16),
        mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N),
    )


def _triton_gemv(A_vec, B_mat, out_shape):
    """
    A_vec: [M_tokens, K], bfloat16 tensor
    B_mat: [K, N], bfloat16 tensor
    Returns out: [M_tokens, N], bfloat16 tensor
    """
    M_tokens = A_vec.shape[0]
    K = A_vec.shape[1]
    N = B_mat.shape[1]
    A_vec_c = A_vec.contiguous()
    B_mat_c = B_mat.contiguous()
    Out = torch.empty(out_shape, device=A_vec.device, dtype=torch.bfloat16)
    grid = (M_tokens,)
    matmul_bf16_fp32_kernel[grid](
        A_vec_c, B_mat_c, Out,
        M_tokens, K, N,
        A_vec_c.stride(0), B_mat_c.stride(0), B_mat_c.stride(1),
        Out.stride(0), Out.stride(1),
        BLOCK_K=128,
        num_warps=4,
        num_stages=2,
    )
    return Out


def _triton_matmul(A, B, out_shape):
    """
    A: [M, K], B: [K, N], returns C: [M, N], bfloat16
    """
    M = A.shape[0]
    K = A.shape[1]
    N = B.shape[1]
    A_c = A.contiguous()
    B_c = B.contiguous()
    C = torch.empty(out_shape, device=A.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    matmul_bf16_fp32_kernel[grid](
        A_c, B_c, C,
        M, N, K,
        A_c.stride(0), A_c.stride(1),
        B_c.stride(0), B_c.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        num_warps=4,
        num_stages=2,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(
        self,
        grad_output: torch.Tensor,  # [batch_seq_len, hidden_size] bf16
        hidden_states: torch.Tensor,  # [batch_seq_len, hidden_size] bf16
        router_weight: torch.Tensor,  # [N_experts, hidden_size] bf16
        e_score_correction_bias: torch.Tensor,  # [N_experts] float32
        router_logits: torch.Tensor,  # [batch_seq_len, N_experts] float32
        scores: torch.Tensor,  # [batch_seq_len, N_experts] float32
        topk_indices: torch.Tensor,  # [batch_seq_len, num_experts_per_tok] int64
        topk_weights: torch.Tensor,  # [batch_seq_len, num_experts_per_tok] float32
        score_mask: torch.Tensor,  # [batch_seq_len, N_experts] float32
        shared_expert_gate_weight: torch.Tensor,  # [intermediate_size, hidden_size] bf16
        shared_expert_up_weight: torch.Tensor,  # [intermediate_size, hidden_size] bf16
        shared_expert_down_weight: torch.Tensor,  # [hidden_size, intermediate_size] bf16
        shared_gate_output: torch.Tensor,  # [batch_seq_len, intermediate_size] bf16
        shared_up_output: torch.Tensor,  # [batch_seq_len, intermediate_size] bf16
        shared_activated: torch.Tensor,  # [batch_seq_len, hidden_size] bf16
    ):
        """
        Triton-based backward using custom kernels. Returns:
        - grad_hidden_states
        - grad_router_weight
        - grad_shared_expert_gate_weight
        - grad_shared_expert_up_weight
        - grad_shared_expert_down_weight
        """
        batch_seq_len = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        N_experts = router_weight.shape[0]
        intermediate_size = shared_expert_gate_weight.shape[0]
        K = shared_expert_gate_weight.shape[1]  # hidden_size

        # 1) Per-token GEMV: grad_hidden_from_shared_gate
        grad_hidden_from_shared_gate = _triton_gemv(
            grad_shared_gate_output, shared_expert_gate_weight, (batch_seq_len, hidden_size)
        )

        # 2) Per-token GEMV: grad_hidden_from_shared_up
        grad_hidden_from_shared_up = _triton_gemv(
            grad_shared_up_output, shared_expert_up_weight, (batch_seq_len, hidden_size)
        )

        # 3) GEMMs for shared_expert_down_weight
        grad_shared_expert_down_weight = _triton_matmul(
            grad_shared_output.t(), shared_activated, (hidden_size, intermediate_size)
        )

        # 4) GEMM for router_weight: grad_router_weight = grad_router_logits.T @ hidden_states
        grad_router_weight = _triton_matmul(
            hidden_states.t(), grad_router_logits, (hidden_size, N_experts)
        )

        # 5) GEMMs for shared_expert_up_weight and shared_expert_gate_weight
        grad_shared_expert_up_weight = _triton_matmul(
            grad_shared_up_output.t(), hidden_states, (intermediate_size, hidden_size)
        )
        grad_shared_expert_gate_weight = _triton_matmul(
            grad_shared_gate_output.t(), hidden_states, (intermediate_size, hidden_size)
        )

        # Sum per-token contributions to hidden states gradient
        grad_hidden_states = grad_hidden_from_shared_gate + grad_hidden_from_shared_up

        # Return gradients
        return (
            grad_hidden_states,
            grad_router_weight,
            grad_shared_expert_gate_weight,
            grad_shared_expert_up_weight,
            grad_shared_expert_down_weight,
        )


def run(*args):
    return ModelNew()(*args)
