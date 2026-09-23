import torch
import triton
import triton.language as tl


# -------------------------
# Triton kernels
# -------------------------

# GEMV: compute y = A @ v where A is [M, K] and v is [K], output y is [M]
# One program per row (M dimension). Vectorized loads along K.
@triton.jit
def triton_gemv_bf16_row(A_ptr, v_ptr, y_ptr, M, K):
    row = tl.program_id(0)
    acc = 0.0
    # Iterate over K in chunks of BK (128)
    for k0 in range(0, K, 128):
        offs_k = k0 + tl.arange(0, 128)
        a = tl.load(A_ptr + row * K + offs_k, mask=offs_k < K, other=0.0)
        b = tl.load(v_ptr + offs_k, mask=offs_k < K, other=0.0)
        acc += tl.sum(a.to(tl.float32) * b.to(tl.float32), axis=0)
    tl.store(y_ptr + row, acc.to(tl.bfloat16))


# Matmul: compute C = A @ B where A is [M, K], B is [K, N], output C is [M, N] bfloat16
# 2D tiling over M and N, iterate K in chunks of BK, fp32 accumulation.
@triton.jit
def triton_matmul_bf16(A_ptr, B_ptr, C_ptr,
                        M, N, K,
                        A_stride_m, A_stride_k,
                        B_stride_k, B_stride_n,
                        C_stride_m, C_stride_n,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offsets = pid_m * BM + tl.arange(0, BM)
    n_offsets = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k0 in range(0, K, BK):
        k_offsets = k0 + tl.arange(0, BK)
        a_ptrs = A_ptr + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_ptrs = B_ptr + k_offsets[:, None] * B_stride_k + n_offsets[None, :] * B_stride_n
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    c_ptrs = C_ptr + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


# -------------------------
# ModelNew: Triton-backed forward
# -------------------------

class ModelNew(torch.nn.Module):
    def forward(self, grad_output, hidden_states, router_weight,
                e_score_correction_bias,
                router_logits, scores, topk_indices, topk_weights, score_mask,
                shared_expert_gate_weight, shared_expert_up_weight, shared_expert_down_weight,
                shared_gate_output, shared_up_output, shared_activated):
        # Shapes
        hidden_size = shared_expert_gate_weight.shape[1]
        batch_seq_len = hidden_states.shape[0]
        intermediate_size = shared_expert_up_weight.shape[0]
        n_routed_experts = router_weight.shape[0]

        # Ensure contiguity for Triton
        grad_output = grad_output.contiguous()
        hidden_states = hidden_states.contiguous()
        shared_expert_gate_weight = shared_expert_gate_weight.contiguous()
        shared_expert_up_weight = shared_expert_up_weight.contiguous()
        shared_expert_down_weight = shared_expert_down_weight.contiguous()
        shared_gate_output = shared_gate_output.contiguous()
        shared_up_output = shared_up_output.contiguous()
        # Pass-through tensors (contiguity not required for math here)
        router_weight = router_weight.contiguous()
        e_score_correction_bias = e_score_correction_bias.contiguous()
        topk_indices = topk_indices.contiguous()
        topk_weights = topk_weights.contiguous()
        score_mask = score_mask.contiguous()

        # 1) Per-token GEMVs:
        # grad_hidden_from_shared_up[token] = grad_shared_up_output[token] @ shared_expert_up_weight
        # grad_hidden_from_shared_gate[token] = grad_shared_gate_output[token] @ shared_expert_gate_weight
        grad_hidden_from_shared_up = torch.empty((batch_seq_len, hidden_size), dtype=torch.bfloat16, device=hidden_states.device)
        grad_hidden_from_shared_gate = torch.empty((batch_seq_len, hidden_size), dtype=torch.bfloat16, device=hidden_states.device)

        # Launch Triton GEMV: one program per token
        for token in range(batch_seq_len):
            # For up: A[token, :] = grad_shared_up_output[token], v = shared_expert_up_weight
            A_up = shared_up_output[token]  # [hidden_size]
            v_up = shared_expert_up_weight  # [hidden_size, hidden_size] -- we only need columns; use as vector by flattening? Not applicable.
            # Use A_up as [1, hidden_size] view to matmul kernel? Simpler: implement G


def run(*args):
    return ModelNew()(*args)
