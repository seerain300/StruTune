# task: SOL/L2/030_flux_concatenated_sequence_processing_with_split
# batch: stts
# pass_at_1: 0.25
# final_geomean_speedup(A800, official re-eval): 0.143
# provenance: sample s3 reward=0.715 feedback_geomean=0.149 turn=42
import torch
import triton
import triton.language as tl


@triton.jit
def _batched_matmul_singlepass(
    A_ptr,  # [B, M, K]
    B_ptr,  # [K, N]
    C_ptr,  # [B, M, N]
    B, M, N, K,
    stride_A_b, stride_A_m, stride_A_k,
    stride_B_k, stride_B_n,
    stride_C_b, stride_C_m, stride_C_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # program ids
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # tile offsets
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # masks for boundaries
    m_mask = m_offsets < M
    n_mask = n_offsets < N

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # single-pass reduction over K
    for k in range(0, K):
        # load A[b, m, k] -> [BLOCK_M]
        a_ptrs = A_ptr + pid_b * stride_A_b + m_offsets * stride_A_m + k * stride_A_k
        a = tl.load(a_ptrs, mask=m_mask, other=0.0)
        a = a.to(tl.float32)  # accumulate in fp32

        # load B[k, n] -> [BLOCK_N]
        b_ptrs = B_ptr + k * stride_B_k + n_offsets * stride_B_n
        b = tl.load(b_ptrs, mask=n_mask, other=0.0)
        b = b.to(tl.float32)

        # outer product accumulate
        acc += a[:, None] * b[None, :]

    # write back C[b, m, n] = acc
    c_ptrs = C_ptr + pid_b * stride_C_b + m_offsets[:, None] * stride_C_m + n_offsets[None, :] * stride_C_n
    # combined mask for store
    mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, acc, mask=mask)


def _triton_batched_gemm(A, B, BLOCK_M=64, BLOCK_N=64):
    """
    A: [B, M, K] (e.g., encoder_hidden_states: [B, T, D])
    B: [K, N] (e.g., process_weight.T: [D, D])
    Returns C: [B, M, N] via Triton.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    B_shape = B.shape
    assert B.ndim == 2 and B.shape[0] == A.shape[2], "B must be [K, N] with K = hidden_dim"
    B_size = B_shape[1]
    A = A.contiguous()
    B = B.contiguous()

    Bsz, M, K = A.shape
    C = torch.empty((Bsz, M, B_size), device=A.device, dtype=torch.float32)

    grid = (Bsz, triton.cdiv(M, BLOCK_M), triton.cdiv(B_size, BLOCK_N))
    _batched_matmul_singlepass[grid](
        A, B, C,
        Bsz, M, B_size, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=1,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, D]
        encoder_hidden_states: [B, T, D]
        process_weight: [D, D]
        returns (processed_encoder: [B, T, D], processed_hidden: [B, I, D])
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be CUDA"
        Bsz = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert process_weight.shape == (D, D), "process_weight must be [hidden_dim, hidden_dim]"

        # Triton GEMM for encoder stream: [B, T, D] @ [D, D] -> [B, T, D]
        processed_encoder = _triton_batched_gemm(encoder_hidden_states, process_weight.t())

        # Triton GEMM for hidden stream: [B, I, D] @ [D, D] -> [B, I, D]
        processed_hidden = _triton_batched_gemm(hidden_states, process_weight.t())

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
