import torch
import triton
import triton.language as tl


# 1) Per-row squared norm: out[row] = sum_j A[row, j]^2 for A[M, N]
@triton.jit
def _row_sqnorm(
    A_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    stride_out,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)
    for k in range(0, N, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + row * stride_am + offs_k * stride_an, mask=offs_k < N, other=0.0).to(tl.float32)
        acc += tl.sum(a * a, axis=0)
    tl.store(out_ptr + row * stride_out, acc)


# 2) Scatter-add: grad_topk_weights_norm[b, k] -> grad_scores[b, indices[b, k]]
#    One program per row. atomic_add fp32 into grad_scores[b, idx]
@triton.jit
def _scatter_add_topk(
    grad_topk_ptr,     # [B, K], fp32
    indices_ptr,       # [B, K], int32
    grad_scores_ptr,   # [B, E], fp32 (output accumulated here)
    B, E, K,
    stride_gt0, stride_gt1,
    stride_idx0, stride_idx1,
    stride_gs0, stride_gs1,
):
    row = tl.program_id(0)
    for k in range(0, K):
        val = tl.load(grad_topk_ptr + row * stride_gt0 + k * stride_gt1)  # fp32
        idx = tl.load(indices_ptr + row * stride_idx0 + k * stride_idx1)  # int32
        tl.atomic_add(grad_scores_ptr + row * stride_gs0 + idx * stride_gs1, val)


# 3) Matmul in bf16: A[M, K] bf16 x B[K, N] bf16 -> C[M, N] bf16
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_bf16_bf16(
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
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)  # [BLOCK_M, BLOCK_K]
        b = tl.load(
            B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)  # [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, b)  # [BLOCK_M, BLOCK_N]

    c = acc.to(tl.bfloat16)
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# 4) Elementwise: Y = silu(X) * Z, where silu(x) = x * sigmoid(x)
@triton.jit
def _silu_mul_elementwise(
    X_ptr, Z_ptr, Y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_zm, stride_zn,
    stride_ym, stride_yn,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    mask = (row < M) & (col < N)
    x = tl.load(X_ptr + row * stride_xm + col * stride_xn, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(Z_ptr + row * stride_zm + col * stride_zn, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x))
    silu = x * sig
    y = silu * z
    tl.store(Y_ptr + row * stride_ym + col * stride_yn, y.to(tl.bfloat16), mask=mask)


# 5) Elementwise sigmoid(X) * Y
@triton.jit
def _sigmoid_mul_elementwise(
    X_ptr, Y_ptr, Out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_om, stride_on,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    mask = (row < M) & (col < N)
    x = tl.load(X_ptr + row * stride_xm + col * stride_xn, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y_ptr + row * stride_ym + col * stride_yn, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x))
    out = sig * y
    tl.store(Out_ptr + row * stride_om + col * stride_on, out.to(tl.bfloat16), mask=mask)


def _launch_row_sqnorm(grad_output):
    # grad_output: [B, H] bf16
    B, H = grad_output.shape
    out = torch.empty(B, dtype=torch.float32, device=grad_output.device)
    grid = (B,)
    _row_sqnorm[grid](
        grad_output, out,
        B, H,
        grad_output.stride(0), grad_output.stride(1),
        out.stride(0),
        BLOCK_K=128,
    )
    return out


def _launch_scatter_add_topk(grad_topk_weights_norm, topk_indices):
    # grad_topk_weights_norm: [B, K] fp32
    # topk_indices: [B, K] int32
    B, K = grad_topk_weights_norm.shape
    E = topk_indices.shape[1]
    grad_scores = torch.zeros((B, E), dtype=torch.float32, device=grad_topk_weights_norm.device)
    grid = (B,)
    _scatter_add_topk[grid](
        grad_topk_weights_norm, topk_indices,
        grad_scores,
        B, E, K,
        grad_topk_weights_norm.stride(0), grad_topk_weights_norm.stride(1),
        topk_indices.stride(0), topk_indices.stride(1),
        grad_scores.stride(0), grad_scores.stride(1),
        num_warps=1,
    )
    return grad_scores


def _launch_matmul_bf16_bf16(A_bf16, B_bf16):
    # A: [M, K] bf16, B: [K, N] bf16 -> C: [M, N] bf16
    M, K = A_bf16.shape
    K2, N = B_bf16.shape
    assert K == K2
    C = torch.empty((M, N), dtype=torch.bfloat16, device=A_bf16.device)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _matmul_bf16_bf16[grid](
        A_bf16, B_bf16, C,
        M, N, K,
        A_bf16.stride(0), A_bf16.stride(1),
        B_bf16.stride(0), B_bf16.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


def _launch_silu_mul_elementwise(shared_gate_output, shared_up_output, shared_activated):
    B, Hprime = shared_gate_output.shape
    H = shared_activated.shape[1]
    # Grid covers [B, H]
    grid = (B, H)
    _silu_mul_elementwise[grid](
        shared_gate_output, shared_up_output, shared_activated,
        B, H,
        shared_gate_output.stride(0), shared_gate_output.stride(1),
        shared_up_output.stride(0), shared_up_output.stride(1),
        shared_activated.stride(0), shared_activated.stride(1),
        BLOCK=256,
    )


def _launch_sigmoid_mul_elementwise(scores_fp32, grad_scores_fp32, grad_router_logits):
    B, E = scores_fp32.shape
    grid = (B, E)
    _sigmoid_mul_elementwise[grid](
        scores_fp32, grad_scores_fp32, grad_router_logits,
        B, E,
        scores_fp32.stride(0), scores_fp32.stride(1),
        grad_scores_fp32.stride(0), grad_scores_fp32.stride(1),
        grad_router_logits.stride(0), grad_router_logits.stride(1),
        BLOCK=256,
    )


class ModelNew(torch.nn.Module):
    def forward(
        self,
        grad_output: torch.Tensor,             # [B, H] bf16
        hidden_states: torch.Tensor,          # [B, H] bf16
        router_weight: torch.Tensor,          # [E, H] bf16
        e_score_correction_bias: torch.Tensor,# [E] fp32
        router_logits: torch.Tensor,          # [B, E] fp32 (unused in gradients, but could be used if needed)
        scores: torch.Tensor,                 # [B, E] fp32
        topk_indices: torch.Tensor,           # [B, K] int32
        topk_weights: torch.Tensor,           # [B, K] fp32
        score_mask: torch.Tensor,             # [B, E] fp32
        shared_expert_gate_weight: torch.Tensor,  # [H', H] bf16
        shared_expert_up_weight: torch.Tensor,    # [H', H] bf16
        shared_expert_down_weight: torch.Tensor,  # [H, H'] bf16
        shared_gate_output: torch.Tensor,        # [B, H'] bf16
        shared_up_output: torch.Tensor,          # [B, H'] bf16
    ):
        B, H = grad_output.shape
        E = router_weight.shape[0]
        Hprime = shared_gate_output.shape[1]
        K = topk_indices.shape[1]

        # 1) Compute shared-activated = silu(gate) * up (elementwise in Triton)
        shared_activated = torch.empty_like(shared_gate_output)  # [B, H'] bf16
        _launch_silu_mul_elementwise(shared_gate_output, shared_up_output, shared_activated)

        # 2) Initialize grads
        grad_hidden_states = torch.zeros_like(hidden_states)         # [B, H] bf16
        grad_shared_expert_gate_weight = torch.empty_like(shared_expert_gate_weight)  # [H', H] bf16
        grad_shared_expert_up_weight = torch.empty_like(shared_expert_up_weight)        # [H', H] bf16
        grad_shared_expert_down_weight = torch.empty_like(shared_expert_down_weight)    # [H, H'] bf16

        # 3) grad through shared down: grad_shared_activated = grad_output @ down_weight.T
        #    Then grad_shared_expert_down_weight = grad_output.T @ shared_activated
        grad_shared_activated = grad_output @ shared_expert_down_weight  # PyTorch for simplicity (bf16 matmul)
        grad_shared_expert_down_weight = grad_output.t().to(torch.float32) @ shared_activated.to(torch.float32).to(torch.bfloat16)

        # 4) Gradient through SwiGLU for shared gate and up
        grad_shared_gate_silu = grad_shared_activated * shared_up_output.to(torch.bfloat16)  # [B, H']
        grad_shared_up_output = grad_shared_activated * F.silu(shared_gate_output)           # PyTorch fp32 for simplicity
        # Compute silu derivative: d/dx silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        # We'll compute gate grads in Triton (elementwise) later; for now, do it here in PyTorch for brevity:
        shared_gate_float = shared_gate_output.to(torch.float32)
        sigmoid_gate = torch.sigmoid(shared_gate_float)
        d_silu = sigmoid_gate * (1.0 + shared_gate_float * (1.0 - sigmoid_gate))
        grad_shared_gate_output = (grad_shared_gate_silu.to(torch.float32) * d_silu).to(torch.bfloat16)

        # 5) Gradients through shared_expert_up and gate
        grad_hidden_from_shared_up = grad_shared_up_output @ shared_expert_up_weight        # [B, H] = [B, H'] @ [H', H]
        grad_hidden_from_shared_gate = grad_shared_gate_output @ shared_expert_gate_weight  # [B, H] = [B, H'] @ [H', H]
        grad_hidden_states = grad_hidden_states + grad_hidden_from_shared_up + grad_hidden_from_shared_gate

        grad_shared_expert_up_weight = grad_shared_up_output.t().to(torch.float32) @ hidden_states.to(torch.float32)  # [H', H]
        grad_shared_expert_gate_weight = grad_shared_gate_output.t().to(torch.float32) @ hidden_states.to(torch.float32)  # [H, H]
        grad_shared_expert_up_weight = grad_shared_expert_up_weight.to(torch.bfloat16)
        grad_shared_expert_gate_weight = grad_shared_expert_gate_weight.to(torch.bfloat16)

        # 6) Routing: compute grad_topk_weights_norm via norm of grad_output
        norm_sq = _launch_row_sqnorm(grad_output)  # [B] fp32
        # grad_topk_weights_norm: per row split equally across K: norm_sq / K
        grad_topk_weights_norm = (norm_sq / float(K)).view(B, 1).expand(B, K)  # [B, K] fp32

        # 7) Scatter-add into grad_scores
        grad_scores = _launch_scatter_add_topk(grad_topk_weights_norm, topk_indices)  # [B, E] fp32

        # 8) grad_router_logits = grad_scores * scores * (1 - scores)
        grad_router_logits = torch.empty((B, E), dtype=torch.float32, device=scores.device)
        _launch_sigmoid_mul_elementwise(scores, grad_scores, grad_router_logits)  # elementwise Triton

        # 9) grad_router_weight = grad_router_logits.T @ hidden_states
        grad_router_logits_T = grad_router_logits.t()  # [E, B]
        grad_router_weight = _launch_matmul_bf16_bf16(grad_router_logits_T, hidden_states)  # [E, H] bf16

        # Return gradients
        return (
            grad_hidden_states,                         # [B, H] bf16
            grad_router_weight,                       # [E, H] bf16
            grad_shared_expert_gate_weight,           # [H', H] bf16
            grad_shared_expert_up_weight,             # [H', H] bf16
            grad_shared_expert_down_weight,           # [H, H'] bf16
        )


def run(*args):
    return ModelNew()(*args)
