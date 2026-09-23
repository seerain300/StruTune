import torch
import triton
import triton.language as tl


# Triton matmul kernel: C = A @ B
# A: [M, K], B: [K, N], C: [M, N] bfloat16
@triton.jit
def triton_matmul_bf16(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    A_stride_m, A_stride_k,
    B_stride_k, B_stride_n,
    C_stride_m, C_stride_n,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BM + tl.arange(0, BM)
    n_offsets = pid_n * BN + tl.arange(0, BN)

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k0 in range(0, K, BK):
        k_offsets = k0 + tl.arange(0, BK)
        a = tl.load(
            A_ptr + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k,
            mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B_ptr + k_offsets[:, None] * B_stride_k + n_offsets[None, :] * B_stride_n,
            mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    tl.store(
        C_ptr + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n,
        acc.to(tl.bfloat16),
        mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N),
    )


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args correspond to the original run signature
        # Return a dict mirroring the original outputs, including Triton-computed heavy matrices.

        # Extract devices and shapes
        device = args[0].device
        grad_output = args[0]  # [batch_seq_len, hidden_size]
        hidden_states = args[1]  # [batch_seq_len, hidden_size]
        router_weight = args[2]  # [N_experts, hidden_size]
        e_score_correction_bias = args[3]  # [N_experts] float32
        router_logits = args[4]  # [batch_seq_len, N_experts]
        scores = args[5]  # [batch_seq_len, N_experts]
        topk_indices = args[6]  # [batch_seq_len, num_experts_per_tok]
        topk_weights = args[7]  # [batch_seq_len, num_experts_per_tok]
        score_mask = args[8]  # [batch_seq_len, N_experts]
        shared_expert_gate_weight = args[9]  # [intermediate_size, hidden_size]
        shared_expert_up_weight = args[10]  # [intermediate_size, hidden_size]
        shared_expert_down_weight = args[11]  # [hidden_size, intermediate_size] (unused in our computation)
        shared_gate_output = args[12]  # [batch_seq_len, hidden_size]
        shared_up_output = args[13]  # [batch_seq


def run(*args):
    return ModelNew()(*args)
