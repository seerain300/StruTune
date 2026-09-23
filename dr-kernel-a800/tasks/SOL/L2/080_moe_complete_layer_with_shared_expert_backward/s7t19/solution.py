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
        val = tl.load(grad_topk_weights_ptr + row * stride_gtopk0 + k * stride_gtopk1)  # fp32
        idx = tl.load(indices_ptr + row * stride_idx0 + k * stride_idx1)                # int32
        # atomic add into grad_scores[row, idx]
        tl.atomic_add(grad_scores_ptr + row * stride_gscore0 + idx * stride_gscore1, val)


# Triton matmul kernel: A[M, K] bf16 x B[K, N] bf16 -> C[M, N] bf16 (fp32 accumulation)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8),
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
            other=0.0
        ).to(tl.float16)
        b = tl.load(
            B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        ).to(tl.float16)
        acc += tl.dot(a, b)  # fp32 accumulation
    c = acc.to(tl.float16)
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# Triton elementwise: silu(x) = x * sigmoid(x)
@triton.jit
def _silu_elementwise(
    x_ptr, y_ptr,
    M, N,
    stride_x0, stride_x1,
    stride_y0, stride_y1,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    # load x[row, col] and compute silu(x) = x * sigmoid(x)
    x = tl.load(x_ptr + row * stride_x0 + col * stride_x1, mask=(row < M) & (col < N), other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + row * stride_y0 + col * stride_y1, y, mask=(row < M) & (col < N))


# Triton elementwise: grad = grad_scores * scores * (1 - scores)
@triton.jit
def _sigmoid_mul_elementwise(
    scores_ptr, grad_scores_ptr, grad_ptr,
    M, N,
    stride_s0, stride_s1,
    stride_gs0, stride_gs1,
    stride_g0, stride_g1,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    scores = tl.load(scores_ptr + row * stride_s0 + col * stride_s1, mask=(row < M) & (col < N), other=0.0).to(tl.float32)
    grad_scores = tl.load(grad_scores_ptr + row * stride_gs0 + col * stride_gs1, mask=(row < M) & (col < N), other=0.0).to(tl.float32)
    # compute gradient w.r.t. scores: grad = grad_scores * sigmoid(scores) * (1 - sigmoid(scores))
    sig = 1.0 / (1.0 + tl.exp(-scores))
    grad = grad_scores * sig * (1.0 - sig)
    tl.store(grad_ptr + row * stride_g0 + col * stride_g1, grad, mask=(row < M) & (col < N))


def _launch_matmul_bf16_bf16(A_bf16, B_bf16, C_bf16):
    M, K = A_bf16.shape
    K2, N = B_bf16.shape
    assert K == K2
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _matmul_bf16_bf16[grid](
        A_bf16, B_bf16, C_bf16,
        M, N, K,
        A_bf16.stride(0), A_bf16.stride(1),
        B_bf16.stride(0), B_bf16.stride(1),
        C_bf16.stride(0), C_bf16.stride(1),
    )


def _launch_silu_elementwise(x_bf16, y_fp32, M, N, BLOCK=256):
    # y is fp32 output; we cast back to bf16 later
    grid = (M, N)
    _silu_elementwise[grid](
        x_bf16, y_fp32,
        M, N,
        x_bf16.stride(0), x_bf16.stride(1),
        y_fp32.stride(0), y_fp32.stride(1),
        BLOCK=BLOCK,
    )


def _launch_sigmoid_mul_elementwise(scores_fp32, grad_scores_fp32, grad_router_logits_fp32, M, N, BLOCK=256):
    grid = (M, N)
    _sigmoid_mul_elementwise[grid](
        scores_fp32, grad_scores_fp32, grad_router_logits_fp32,
        M, N,
        scores_fp32.stride(0), scores_fp32.stride(1),
        grad_scores_fp32.stride(0), grad_scores_fp32.stride(1),
        grad_router_logits_fp32.stride(0), grad_router_logits_fp32.stride(1),
        BLOCK=BLOCK,
    )


class ModelNew(torch.nn.Module):
    def forward(
        self,
        grad_output: torch.Tensor,             # [B, H] bf16
        hidden_states: torch.Tensor,          # [B, H] bf16
        router_weight: torch.Tensor,          # [E, H] bf16
        e_score_correction_bias: torch.Tensor,# [E] fp32 (unused here)
        router_logits: torch.Tensor,          # [B, E] fp32 (unused here)
        scores: torch.Tensor,                 # [B, E] fp32
        topk_indices: torch.Tensor,           # [B, K] int32
        topk_weights: torch.Tensor,           # [B, K] fp32 (post-scaling and normalization)
        score_mask: torch.Tensor,             # [B, E] fp32 (not used in this routing logic)
        shared_expert_gate_weight: torch.Tensor,  # [Hprime, H] bf16
        shared_expert_up_weight: torch.Tensor,    # [Hprime, H] bf16
        shared_expert_down_weight: torch.Tensor,  # [H, Hprime] bf16
        shared_gate_output: torch.Tensor,         # [B, Hprime] bf16
        shared_up_output: torch.Tensor,           # [B, Hprime] bf16
    ):
        # Ensure tensors are on CUDA and contiguous
        device = grad_output.device
        assert device.type == "cuda", "ModelNew.forward requires CUDA tensors"
        # 1) Compute shared_activated = silu(shared_gate_output) * shared_up_output (elementwise)
        shared_activated_fp32 = torch.empty((shared_gate_output.shape[0], shared_gate_output.shape[1]), dtype=torch.float32, device=device)
        _launch_silu_elementwise(
            shared_gate_output, shared_activated_fp32,
            shared_gate_output.shape[0], shared_gate_output.shape[1],
        )

        # 2) Gradients through shared paths
        # a) grad_shared_expert_down_weight = grad_output.T [B, H] x shared_activated [H, H'] -> [H, H']
        grad_output_t_bf16 = grad_output.transpose(0, 1).contiguous()  # [H, B]
        shared_activated_t_bf16 = shared_activated_fp32.t().to(torch.bfloat16).contiguous()  # [H, H'] cast to bf16 for matmul
        grad_shared_expert_down_weight = torch.empty((grad_output.shape[1], shared_expert_down_weight.shape[1]), dtype=torch.bfloat16, device=device)
        _launch_matmul_bf16_bf16(grad_output_t_bf16, shared_activated_t_bf16, grad_shared_expert_down_weight)

        # b) grad_shared_expert_up_weight = grad_shared_up_output.T [H', B] x hidden_states [B, H] -> [H', H]
        grad_shared_up_output_t_bf16 = grad_shared_up_output.transpose(0, 1).contiguous()  # [H', B]
        hidden_states_bf16 = hidden_states.contiguous()  # [B, H]
        grad_shared_expert_up_weight = torch.empty((grad_shared_up_output.shape[1], hidden_states.shape[1]), dtype=torch.bfloat16, device=device)
        _launch_matmul_bf16_bf16(grad_shared_up_output_t_bf16, hidden_states_bf16, grad_shared_expert_up_weight)

        # c) grad_shared_expert_gate_weight = grad_shared_gate_output.T [H, B] x hidden_states [B, H] -> [H, H]
        grad_shared_gate_output_t_bf16 = grad_shared_gate_output.transpose(0, 1).contiguous()  # [H, B]
        grad_shared_expert_gate_weight = torch.empty((grad_shared_gate_output.shape[1], hidden_states.shape[1]), dtype=torch.bfloat16, device=device)
        _launch_matmul_bf16_bf16(grad_shared_gate_output_t_bf16, hidden_states_bf16, grad_shared_expert_gate_weight)

        # d) grad_hidden_from_shared = grad_shared_up_output x shared_expert_down_weight + grad_shared_gate_output x shared_expert_gate_weight
        #    grad_hidden_from_shared_up = grad_shared_up_output [B, H'] x shared_expert_down_weight [H', H] -> [B, H]
        grad_shared_up_output_bf16 = grad_shared_up_output.contiguous()  # [B, H']
        grad_hidden_from_shared_up = torch.empty_like(grad_output, dtype=torch.bfloat16, device=device)
        _launch_matmul_bf16_bf16(grad_shared_up_output_bf16, shared_expert_down_weight.contiguous(), grad_hidden_from_shared_up)

        #    grad_hidden_from_shared_gate = grad_shared_gate_output [B, H] x shared_expert_gate_weight [H, H] -> [B, H]
        grad_shared_gate_output_bf16 = grad_shared_gate_output.contiguous()  # [B, H]
        grad_hidden_from_shared_gate = torch.empty((grad_output.shape[0], grad_output.shape[1]), dtype=torch.bfloat16, device=device)
        _launch_matmul_bf16_bf16(grad_shared_gate_output_bf16, shared_expert_gate_weight.contiguous(), grad_hidden_from_shared_gate)

        grad_hidden_shared = grad_hidden_from_shared_up + grad_hidden_from_shared_gate

        # 3) Route weight gradient:
        #    grad_topk_weights_norm is ||grad_output||^2 / K, per token. Compute it in Triton (row-wise squared norm), then scatter-add into grad_scores.
        grad_output_bf16 = grad_output.contiguous()  # [B, H]
        grad_norm_sq_fp32 = torch.empty((grad_output.shape[0],), dtype=torch.float32, device=device)
        _row_sqnorm[1024](
            grad_output_bf16, grad_norm_sq_fp32,
            grad_output.shape[0], grad_output.shape[1],
            grad_output_bf16.stride(0), grad_output_bf16.stride(1),
            grad_norm_sq_fp32.stride(0),
            H=grad_output.shape[1],
        )
        # grad_topk_weights_norm = grad_norm_sq / K
        grad_norm_sq_expanded = grad_norm_sq_fp32[:, None] / topk_weights.shape[1]
        # Allocate grad_scores [B, E] in fp32 and scatter-add
        grad_scores_fp32 = torch.zeros((grad_output.shape[0], scores.shape[1]), dtype=torch.float32, device=device)
        _scatter_add_topk[(grad_output.shape[0],)](
            grad_norm_sq_expanded, topk_indices, grad_scores_fp32,
            grad_output.shape[0], scores.shape[1], topk_indices.shape[1],
            1, 1,  # strides assumed 1 here since contiguous
            1, 1,  # topk_indices is int32 contiguous
            grad_scores_fp32.stride(0), grad_scores_fp32.stride(1),
        )

        # 4) grad_router_logits = grad_scores * scores * (1 - scores) via Triton elementwise kernel
        grad_router_logits_fp32 = torch.empty_like(scores, dtype=torch.float32, device=device)
        _launch_sigmoid_mul_elementwise(scores, grad_scores_fp32, grad_router_logits_fp32, scores.shape[0], scores.shape[1])

        # 5) grad_router_weight = grad_router_logits.T [E, B] x hidden_states [B, H] -> [E, H]
        grad_router_logits_bf16 = grad_router_logits_fp32.to(torch.bfloat16).transpose(0, 1).contiguous()  # [E, B]
        hidden_states_bf16 = hidden_states.contiguous()  # [B, H]
        grad_router_weight = torch.empty((scores.shape[1], hidden_states.shape[1]), dtype=torch.bfloat16, device=device)
        _launch_matmul_bf16_bf16(grad_router_logits_bf16, hidden_states_bf16, grad_router_weight)

        # 6) Combine shared and routed grads for hidden
        grad_hidden_states = grad_hidden_shared  # [B, H]

        # Return gradients tuple as expected by original: (grad_hidden_states, grad_router_weight, gate, up, down)
        return (
            grad_hidden_states,
            grad_router_weight,
            grad_shared_expert_gate_weight,      # [H, H]
            grad_shared_expert_up_weight,        # [H', H]
            grad_shared_expert_down_weight,      # [H, H']
        )


def run(*args):
    return ModelNew()(*args)
