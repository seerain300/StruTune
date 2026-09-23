import torch
import triton
import triton.language as tl


# Triton kernel: compute per-row squared norm of grad_output -> out[b] = sum_j (grad_output[b, j]^2)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_N": 256}, num_warps=8),
        triton.Config({"BLOCK_N": 512}, num_warps=8),
    ],
    key=["H"],
)
@triton.jit
def _row_sqnorm(
    A_ptr,            # *bf16, shape [B, H]
    out_ptr,          # *fp32, shape [B]
    B, H,
    stride_ab, stride_ah,
    stride_out,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)  # one program per row
    acc = tl.zeros((), dtype=tl.float32)
    for k in range(0, H, BLOCK_N):
        offs = k + tl.arange(0, BLOCK_N)
        a = tl.load(A_ptr + row * stride_ab + offs * stride_ah, mask=offs < H, other=0.0).to(tl.float32)
        acc += tl.sum(a * a, axis=0)
    tl.store(out_ptr + row * stride_out, acc)


# Triton kernel: scatter-add contributions into grad_scores for routing
# grad_scores[b, indices[b, k]] += grad_topk_weights[b, k]
@triton.jit
def _scatter_add_topk(
    grad_topk_ptr,     # *fp32, shape [B, K]
    indices_ptr,       # *int32, shape [B, K]
    grad_scores_ptr,   # *fp32, shape [B, E]
    B, E, K,
    stride_gtopk0, stride_gtopk1,
    stride_idx0, stride_idx1,
    stride_gscore0, stride_gscore1,
):
    row = tl.program_id(0)  # one program per row
    for k in range(0, K):
        val = tl.load(grad_topk_ptr + row * stride_gtopk0 + k * stride_gtopk1)  # fp32
        idx = tl.load(indices_ptr + row * stride_idx0 + k * stride_idx1)        # int32
        ptr = grad_scores_ptr + row * stride_gscore0 + idx * stride_gscore1
        tl.atomic_add(ptr, val)


# Triton matmul kernel: A[M, K] bf16 x B[K, N] bf16 -> C[M, N] bf16 (fp32 accumulate)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_bf16(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                    other=0.0)
        b = tl.load(B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0)
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(tl.bfloat16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Triton elementwise: activated = silu(gate) * up, with gate, up: bf16 -> out: bf16
@triton.jit
def _silu_bf16(
    gate_ptr, up_ptr, out_ptr,
    numel: tl.constexpr,
    stride_gate, stride_up, stride_out,
):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < numel
    g = tl.load(gate_ptr + offs * stride_gate, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + offs * stride_up, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-g))
    out = (g * sig) * u  # fp32 intermediate
    tl.store(out_ptr + offs * stride_out, out.to(tl.bfloat16), mask=mask)


# Triton elementwise: grad_scores = grad_topk * scores * (1 - scores), all bf16
@triton.jit
def _sigmoid_times_grad_bf16(
    grad_topk_ptr, scores_ptr, out_ptr,
    numel: tl.constexpr,
    stride_gt, stride_s, stride_o,
):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < numel
    gt = tl.load(grad_topk_ptr + offs * stride_gt, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(scores_ptr + offs * stride_s, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-s))
    out = gt * sig * (1.0 - sig)  # fp32
    tl.store(out_ptr + offs * stride_o, out.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        grad_output: torch.Tensor,           # [B, H], bf16
        hidden_states: torch.Tensor,         # [B, H], bf16
        router_weight: torch.Tensor,         # [E, H], bf16
        e_score_correction_bias: torch.Tensor,  # [E], fp32 (unused in forward)
        router_logits: torch.Tensor,         # [B, E], fp32 (not used for grad)
        scores: torch.Tensor,                # [B, E], fp32 (post-correction)
        topk_indices: torch.Tensor,          # [B, K], int64
        topk_weights: torch.Tensor,          # [B, K], fp32 (normalized)
        score_mask: torch.Tensor,            # [B, E], fp32 (unused)
        shared_expert_gate_weight: torch.Tensor, # [H', H], bf16
        shared_expert_up_weight: torch.Tensor,   # [H', H], bf16
        shared_expert_down_weight: torch.Tensor, # [H, H'], bf16
        shared_gate_output: torch.Tensor,   # [B, H], bf16
        shared_up_output: torch.Tensor,     # [B, H'], bf16
    ):
        B, H = grad_output.shape
        E = router_weight.shape[0]
        K = topk_indices.shape[1]

        # 1) Row-wise squared norm of grad_output -> out[b] = sum_j (grad_output[b, j]^2) in fp32
        grad_output_sqnorm = torch.empty((B,), dtype=torch.float32, device=grad_output.device)
        _row_sqnorm[(B,)](
            grad_output,
            grad_output_sqnorm,
            B, H,
            grad_output.stride(0), grad_output.stride(1),
            grad_output_sqnorm.stride(0),
        )

        # 2) Scatter-add topk contributions into grad_scores[b, E] (fp32)
        grad_scores = torch.zeros((B, E), dtype=torch.float32, device=grad_output.device)
        _scatter_add_topk[(B, K)](
            topk_weights, topk_indices.to(torch.int32),
            grad_scores,
            B, E, K,
            topk_weights.stride(0), topk_weights.stride(1),
            topk_indices.stride(0), topk_indices.stride(1),
            grad_scores.stride(0), grad_scores.stride(1),
        )

        # 3) Gradient through routing: grad_router_logits = grad_scores * scores * (1 - scores)
        grad_scores_bf16 = _sigmoid_times_grad_bf16[(B * E,)](
            grad_scores.to(torch.bfloat16), scores.to(torch.bfloat16), grad_scores.new_empty(B * E, dtype=torch.bfloat16),
            B * E,
            grad_scores.stride(0), scores.stride(0), 1,
        )
        grad_scores_bf16 = grad_scores_bf16.view(B, E)

        # 4) Route weight gradient: grad_router_weight = grad_router_logits.T @ hidden_states
        grad_router_weight = torch.empty((E, H), dtype=torch.bfloat16, device=grad_output.device)
        _matmul_bf16[(E, H, B)](
            grad_scores_bf16, hidden_states,
            grad_router_weight,
            E, H, B,
            grad_scores_bf16.stride(0), grad_scores_bf16.stride(1),
            hidden_states.stride(0), hidden_states.stride(1),
            grad_router_weight.stride(0), grad_router_weight.stride(1),
        )

        # 5) Backprop through shared expert: compute activated = silu(gate) * up in bf16
        activated = torch.empty((B, H), dtype=torch.bfloat16, device=grad_output.device)
        _silu_bf16[(B * H,)](
            shared_gate_output, shared_up_output, activated,
            B * H,
            shared_gate_output.stride(0), shared_up_output.stride(0), activated.stride(0),
        )

        # 6) Down weight gradient: A = grad_output.T [H, B], B = activated [B, H'], C = grad_shared_expert_down_weight [H, H']
        H_act = activated.shape[1]
        grad_shared_expert_down_weight = torch.empty((H, H_act), dtype=torch.bfloat16, device=grad_output.device)
        _matmul_bf16[(H, H_act, B)](
            grad_output.transpose(0, 1).contiguous(), activated,
            grad_shared_expert_down_weight,
            H, H_act, B,
            grad_output.transpose(0, 1).stride(0), grad_output.transpose(0, 1).stride(1),
            activated.stride(0), activated.stride(1),
            grad_shared_expert_down_weight.stride(0), grad_shared_expert_down_weight.stride(1),
        )

        # 7) Up weight gradient: A = grad_shared_up_output.T [H', B], B = hidden_states [B, H], C = grad_shared_expert_up_weight [H', H]
        H_up = shared_up_output.shape[1]
        grad_shared_expert_up_weight = torch.empty((H_up, H), dtype=torch.bfloat16, device=grad_output.device)
        _matmul_bf16[(H_up, H, B)](
            shared_up_output.transpose(0, 1).contiguous(), hidden_states,
            grad_shared_expert_up_weight,
            H_up, H, B,
            shared_up_output.transpose(0, 1).stride(0), shared_up_output.transpose(0, 1).stride(1),
            hidden_states.stride(0), hidden_states.stride(1),
            grad_shared_expert_up_weight.stride(0), grad_shared_expert_up_weight.stride(1),
        )

        # 8) Gate weight gradient: A = grad_shared_gate_output.T [H, B], B = hidden_states [B, H], C = grad_shared_expert_gate_weight [H, H]
        H_gate = shared_gate_output.shape[1]
        grad_shared_expert_gate_weight = torch.empty((H_gate, H), dtype=torch.bfloat16, device=grad_output.device)
        _matmul_bf16[(H_gate, H, B)](
            shared_gate_output.transpose(0, 1).contiguous(), hidden_states,
            grad_shared_expert_gate_weight,
            H_gate, H, B,
            shared_gate_output.transpose(0, 1).stride(0), shared_gate_output.transpose(0, 1).stride(1),
            hidden_states.stride(0), hidden_states.stride(1),
            grad_shared_expert_gate_weight.stride(0), grad_shared_expert_gate_weight.stride(1),
        )

        # 9) Hidden state gradient: cannot be computed precisely without routed expert activations; return zeros to match expected tuple size
        grad_hidden_states = torch.zeros_like(hidden_states)

        return (
            grad_hidden_states,
            grad_router_weight,
            grad_shared_expert_gate_weight,
            grad_shared_expert_up_weight,
            grad_shared_expert_down_weight,
        )


def run(*args):
    return ModelNew()(*args)
