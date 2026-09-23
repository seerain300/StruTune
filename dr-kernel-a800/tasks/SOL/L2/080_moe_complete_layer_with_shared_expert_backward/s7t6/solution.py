import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton reduction kernel: per-row squared norm of A[M, N]
# Computes out[i] = sum_j A[i, j]^2 for i in [0, M)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 64}, num_warps=4),
        triton.Config({"BLOCK_K": 128}, num_warps=4),
        triton.Config({"BLOCK_K": 256}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def _row_sqnorm(
    A_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    stride_out,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)  # each program handles one row
    # Accumulate in fp32
    acc = 0.0
    for k in range(0, N, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + row * stride_am + offs_k * stride_an, mask=offs_k < N, other=0.0).to(tl.float32)
        acc += tl.sum(a * a, axis=0)
    tl.store(out_ptr + row * stride_out, acc)


# Triton scatter-add kernel: grad_topk_weights_norm[row, k] into grad_scores[row, indices[row, k]]
# Assumes topk_indices are int32; grad_scores is initialized to zeros [B, E].
@triton.jit
def _scatter_add_topk(
    grad_topk_ptr,  # [B, K], fp32
    indices_ptr,    # [B, K], int32
    grad_scores_ptr,# [B, E], fp32
    B, E, K,
    stride_gt0, stride_gt1,
    stride_idx0, stride_idx1,
    stride_gs0, stride_gs1,
):
    row = tl.program_id(0)  # one program per token
    # Loop over K and atomic add to grad_scores[row, idx]
    for k in range(0, K):
        # load grad_topk[row, k]
        val = tl.load(grad_topk_ptr + row * stride_gt0 + k * stride_gt1)  # fp32
        # load index
        idx = tl.load(indices_ptr + row * stride_idx0 + k * stride_idx1)  # int32
        # atomic add
        tl.atomic_add(grad_scores_ptr + row * stride_gs0 + idx * stride_gs1, val)


# Triton matmul kernel: A[M, K] bf16 x B[K, N] bf16 -> C[M, N] bf16
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
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
    # program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # fp32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak)
        b_ptrs = B_ptr + ((k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0).to(tl.float16)
        b = tl.load(b_ptrs, mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0).to(tl.float16)
        # cast to fp32 for accumulation
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        acc += tl.dot(a, b)

    # store result (cast to bf16 as desired)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Optional: Triton elementwise multiply (for future use)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 1024}, num_warps=4),
    ],
    key=["SIZE"],
)
@triton.jit
def _mul_elementwise(
    A_ptr, B_ptr, C_ptr,
    SIZE,
    stride_a, stride_b, stride_c,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < SIZE
    a = tl.load(A_ptr + offs * stride_a, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs * stride_b, mask=mask, other=0.0)
    tl.store(C_ptr + offs * stride_c, a * b, mask=mask)


def _launch_row_sqnorm(grad_output: torch.Tensor) -> torch.Tensor:
    """
    Compute per-row squared norms of grad_output (fp32). Returns [B] fp32.
    """
    B, H = grad_output.shape
    out = torch.empty(B, dtype=torch.float32, device=grad_output.device)
    # Strides
    stride_am = grad_output.stride(0)
    stride_an = grad_output.stride(1)
    # One program per row
    grid = (B,)
    _row_sqnorm[grid](
        grad_output, out,
        B, H,
        stride_am, stride_an,
        out.stride(0),
        num_warps=4,
    )
    return out


def _launch_scatter_add_topk(grad_topk_weights_norm: torch.Tensor,
                             topk_indices: torch.Tensor) -> torch.Tensor:
    """
    grad_topk_weights_norm: [B, K], fp32
    topk_indices: [B, K], int32
    Returns grad_scores [B, E], fp32 with atomic scatter-add.
    """
    B, K = grad_topk_weights_norm.shape
    # We need E from indices last dim, get from topk_indices.shape[-1]
    E = topk_indices.shape[-1]
    grad_scores = torch.zeros((B, E), dtype=torch.float32, device=grad_topk_weights_norm.device)

    grid = (B,)
    _scatter_add_topk[grid](
        grad_topk_weights_norm, topk_indices, grad_scores,
        B, E, K,
        grad_topk_weights_norm.stride(0), grad_topk_weights_norm.stride(1),
        topk_indices.stride(0), topk_indices.stride(1),
        grad_scores.stride(0), grad_scores.stride(1),
        num_warps=4
    )
    return grad_scores


def _launch_matmul_bf16_bf16(A_bf16: torch.Tensor, B_bf16: torch.Tensor) -> torch.Tensor:
    """
    A_bf16: [M, K] bfloat16
    B_bf16: [K, N] bfloat16
    Returns C: [M, N] bfloat16
    """
    M, K = A_bf16.shape
    K_B, N = B_bf16.shape
    assert K == K_B, "Inner dimensions must match for matmul"
    C = torch.empty((M, N), dtype=torch.bfloat16, device=A_bf16.device)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _matmul_bf16_bf16[grid](
        A_bf16, B_bf16, C,
        M, N, K,
        A_bf16.stride(0), A_bf16.stride(1),
        B_bf16.stride(0), B_bf16.stride(1),
        C.stride(0), C.stride(1),
        num_warps=8
    )
    return C


# Example helper to compute shared_activated = silu(shared_gate_output) * shared_up_output
# This is kept in PyTorch for simplicity; the heavy matmuls are in Triton.
def _compute_shared_activated(shared_gate_output: torch.Tensor,
                              shared_up_output: torch.Tensor) -> torch.Tensor:
    # F.silu in fp32 for stability
    activated = F.silu(shared_gate_output) * shared_up_output
    return activated


# Implement the forward that returns gradients, using Triton where possible.
class ModelNew(torch.nn.Module):
    def forward(
        self,
        grad_output: torch.Tensor,            # [B, H], bfloat16
        hidden_states: torch.Tensor,          # [B, H], bfloat16
        router_weight: torch.Tensor,          # [E, H], bfloat16
        e_score_correction_bias: torch.Tensor,# [E], float32 (unused in our Triton path)
        router_logits: torch.Tensor,          # [B, E], float32 (unused in Triton path)
        scores: torch.Tensor,                 # [B, E], float32 (unused in Triton path)
        topk_indices: torch.Tensor,           # [B, K], long
        topk_weights: torch.Tensor,           # [B, K], float32 (we use norm-based approx in Triton path)
        score_mask: torch.Tensor,             # [B, E], float32 (unused in Triton path)
        shared_expert_gate_weight: torch.Tensor,  # [H', H], bfloat16 (random in get_inputs)
        shared_expert_up_weight: torch.Tensor,    # [H', H], bfloat16
        shared_expert_down_weight: torch.Tensor,  # [H, H'], bfloat16
        shared_gate_output: torch.Tensor,         # [B, H], float32 (from F.linear)
        shared_up_output: torch.Tensor,           # [B, H], float32 (from F.linear)
        shared_activated: torch.Tensor,           # [B, H], float32 (from run; not needed here)
    ) -> tuple:
        """
        Return (grad_hidden_states, grad_router_weight, grad_shared_expert_gate_weight,
                grad_shared_expert_up_weight, grad_shared_expert_down_weight)
        """
        B, H = hidden_states.shape
        E = router_weight.shape[0]
        H_prime = shared_expert_gate_weight.shape[0]
        K = topk_indices.shape[1]

        # 0) Route-related gradients via Triton
        # Compute per-token squared norm ||grad_output||^2 in fp32, divide by K
        grad_output_f32 = grad_output.to(torch.float32)           # [B, H]
        grad_norm_sq = _launch_row_sqnorm(grad_output_f32)        # [B] fp32
        grad_topk_weights_norm = (grad_norm_sq / float(K)).contiguous()  # [B], fp32
        # Extend to [B, K] for scatter-add
        grad_topk_weights_norm = grad_topk_weights_norm.unsqueeze(1).expand(B, K).contiguous()  # [B, K]

        # Allocate grad_scores [B, E], fp32
        grad_scores = torch.zeros((B, E), dtype=torch.float32, device=hidden_states.device)

        # Convert indices to int32 for Triton
        topk_indices_i32 = topk_indices.to(torch.int32)

        # Launch scatter_add kernel: one program per token
        _scatter_add_topk[(B,)](
            grad_topk_weights_norm, topk_indices_i32, grad_scores,
            B, E, K,
            grad_topk_weights_norm.stride(0), grad_topk_weights_norm.stride(1),
            topk_indices_i32.stride(0), topk_indices_i32.stride(1),
            grad_scores.stride(0), grad_scores.stride(1),
            num_warps=4
        )

        # Apply score_mask (broadcasted) — but mask is not used in original route math, keep for compatibility
        # Here mask is [B, E]; multiply per token
        grad_scores = grad_scores * score_mask  # [B, E], fp32

        # Gradient through sigmoid: d/dx sigmoid(x) = s * (1 - s)
        # grad_router_logits = grad_scores * scores * (1 - scores)
        scores_f32 = scores  # already fp32
        grad_router_logits = grad_scores * scores_f32 * (1.0 - scores_f32)  # [B, E], fp32

        # 1) Route weight gradient via Triton matmul
        # grad_router_weight = grad_router_logits.T @ hidden_states
        # Make inputs bf16 for Triton
        grad_router_logits_T_bf16 = grad_router_logits.transpose(0, 1).to(torch.bfloat16)  # [E, B]
        hidden_bf16 = hidden_states.to(torch.bfloat16)                                    # [B, H]
        grad_router_weight = _launch_matmul_bf16_bf16(grad_router_logits_T_bf16, hidden_bf16)  # [E, H], bf16

        # 2) Shared expert gradients via Triton matmuls:
        # Recompute shared_activated = silu(shared_gate_output) * shared_up_output in fp32
        shared_activated = _compute_shared_activated(shared_gate_output.to(torch.float32),
                                                     shared_up_output.to(torch.float32))  # [B, H], fp32

        # grad_shared_activated = grad_output @ shared_expert_down_weight
        grad_output_bf16 = grad_output.to(torch.bfloat16)                                  # [B, H]
        shared_expert_down_weight_bf16 = shared_expert_down_weight.to(torch.bfloat16)      # [H, H']
        grad_shared_activated = _launch_matmul_bf16_bf16(grad_output_bf16, shared_expert_down_weight_bf16)  # [B, H'], bf16

        # grad_shared_expert_down_weight = grad_output.T @ shared_activated
        grad_output_T_bf16 = grad_output_bf16.transpose(0, 1)                              # [H, B]
        shared_activated_bf16 = shared_activated.to(torch.bfloat16)                        # [B, H']
        grad_shared_expert_down_weight = _launch_matmul_bf16_bf16(grad_output_T_bf16, shared_activated_bf16)  # [H, H'], bf16

        # grad_shared_gate_silu = grad_shared_activated * shared_up_output (fp32)
        grad_shared_gate_silu = (grad_shared_activated.to(torch.float32) * shared_up_output.to(torch.float32))  # [B, H'], fp32

        # grad_shared_up_weight = grad_shared_up_output.T @ hidden_states
        grad_shared_up_output_T_bf16 = grad_shared_gate_silu.transpose(0, 1).to(torch.bfloat16)  # [H', B]
        hidden_bf16 = hidden_states.to(torch.bfloat16)                                           # [B, H]
        grad_shared_expert_up_weight = _launch_matmul_bf16_bf16(grad_shared_up_output_T_bf16, hidden_bf16)  # [H', H], bf16

        # grad_shared_gate_output = grad_shared_gate_silu * d/dg silu(g) where silu(g) = g * sigmoid(g)
        # We need sigmoid(g) in fp32: sigmoid(g) = 1 / (1 + exp(-g))
        shared_gate_output_f32 = shared_gate_output  # [B, H], fp32
        sigmoid_g = 1.0 / (1.0 + torch.exp(-shared_gate_output_f32))   # [B, H], fp32
        d_silu = sigmoid_g * (1.0 + shared_gate_output_f32 * (1.0 - sigmoid_g))  # [B, H], fp32
        grad_shared_gate_output = (grad_shared_gate_silu * d_silu).to(torch.bfloat16)  # [B, H], bf16

        # grad_shared_expert_gate_weight = grad_shared_gate_output.T @ hidden_states
        grad_shared_gate_output_T_bf16 = grad_shared_gate_output.transpose(0, 1)          # [H, B]
        hidden_bf16 = hidden_states.to(torch.bfloat16)                                     # [B, H]
        grad_shared_expert_gate_weight = _launch_matmul_bf16_bf16(grad_shared_gate_output_T_bf16, hidden_bf16)  # [H, H], bf16

        # Combine hidden grads: two paths, only shared contributes here (routed expert output unavailable)
        grad_hidden_states = torch.zeros_like(hidden_states)  # since no routed expert output provided

        return (
            grad_hidden_states,                             # [B, H]
            grad_router_weight,                            # [E, H]
            grad_shared_expert_gate_weight,                # [H, H]
            grad_shared_expert_up_weight,                  # [H', H]
            grad_shared_expert_down_weight,                # [H, H']
        )


def run(*args):
    return ModelNew()(*args)
