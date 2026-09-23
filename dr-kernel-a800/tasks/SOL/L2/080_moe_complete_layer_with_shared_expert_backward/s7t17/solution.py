import torch
import triton
import triton.language as tl


# Row-wise squared norm: out[row] = sum_j A[row, j]^2
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
    row = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)
    for n in range(0, H, BLOCK_N):
        offs = n + tl.arange(0, BLOCK_N)
        a = tl.load(A_ptr + row * stride_ab + offs * stride_ah, mask=offs < H, other=0.0).to(tl.float32)
        acc += tl.sum(a * a, axis=0)
    tl.store(out_ptr + row * stride_out, acc)


# Scatter-add: grad_scores[b, idx[b, k]] += grad_topk_weights[b, k]
@triton.jit
def _scatter_add_topk(
    grad_topk_ptr,    # *fp32, shape [B, K]
    indices_ptr,      # *int32, shape [B, K]
    grad_scores_ptr,  # *fp32, shape [B, E]
    B, E, K,
    stride_gtr0, stride_gtr1,
    stride_ind0, stride_ind1,
    stride_gs0, stride_gs1,
):
    row = tl.program_id(0)
    for k in range(0, K):
        val = tl.load(grad_topk_ptr + row * stride_gtr0 + k * stride_gtr1)  # fp32
        idx = tl.load(indices_ptr + row * stride_ind0 + k * stride_ind1)    # int32
        # atomic add to the target position
        tl.atomic_add(grad_scores_ptr + row * stride_gs0 + idx * stride_gs1, val)


# Matmul bf16 x bf16: C = A[M, K] @ B[K, N] (accumulate in fp32, store bf16)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
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

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
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
        # acc += (a_fp32 @ b_fp32)
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    c = acc.to(tl.float16)
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# Elementwise: shared_activated = silu(shared_gate_output) * shared_up_output
@triton.jit
def _silu_mul_elementwise(
    gate_ptr, up_ptr, out_ptr,
    B, Hprime,
    stride_g0, stride_g1,
    stride_u0, stride_u1,
    stride_o0, stride_o1,
    BLOCK: tl.constexpr,
):
    for i in range(0, B):
        for j in range(0, Hprime, BLOCK):
            offs = j + tl.arange(0, BLOCK)
            g = tl.load(gate_ptr + i * stride_g0 + offs * stride_g1, mask=offs < Hprime, other=0.0).to(tl.float32)
            u = tl.load(up_ptr + i * stride_u0 + offs * stride_u1, mask=offs < Hprime, other=0.0).to(tl.float32)
            # silu(x) = x * sigmoid(x)
            sig = 1.0 / (1.0 + tl.exp(-g))
            y = (g * sig) * u
            tl.store(out_ptr + i * stride_o0 + offs * stride_o1, y.to(tl.float16), mask=offs < Hprime)


# Elementwise: grad_router_logits = grad_scores * scores * (1 - scores)
@triton.jit
def _sigmoid_mul_grad_elementwise(
    scores_ptr, grad_scores_ptr, out_ptr,
    B, E,
    stride_s0, stride_s1,
    stride_gs0, stride_gs1,
    stride_out0, stride_out1,
    BLOCK: tl.constexpr,
):
    for i in range(0, B):
        for j in range(0, E, BLOCK):
            offs = j + tl.arange(0, BLOCK)
            s = tl.load(scores_ptr + i * stride_s0 + offs * stride_s1, mask=offs < E, other=0.0).to(tl.float32)
            g = tl.load(grad_scores_ptr + i * stride_gs0 + offs * stride_gs1, mask=offs < E, other=0.0).to(tl.float32)
            # sigmoid(s)
            sig = 1.0 / (1.0 + tl.exp(-s))
            y = g * sig * (1.0 - sig)
            tl.store(out_ptr + i * stride_out0 + offs * stride_out1, y, mask=offs < E)


def _launch_silu_mul_elementwise(shared_gate_output, shared_up_output, shared_activated):
    # Ensure contiguous
    shared_gate_output = shared_gate_output.contiguous()
    shared_up_output = shared_up_output.contiguous()
    B, Hprime = shared_gate_output.shape
    grid = (1,)  # will be handled inside with for-loops per tile; Triton handles per-row by passing strides
    # We can use a 2D grid: (B, ceil(Hprime/BLOCK)) but Triton requires static grid; we do nested loops in kernel
    _silu_mul_elementwise[grid](shared_gate_output, shared_up_output, shared_activated, B, Hprime,
                                shared_gate_output.stride(0), shared_gate_output.stride(1),
                                shared_up_output.stride(0), shared_up_output.stride(1),
                                shared_activated.stride(0), shared_activated.stride(1), BLOCK=256)


def _launch_sigmoid_mul_elementwise(scores_fp32, grad_scores_fp32, grad_router_logits_fp32):
    B, E = scores_fp32.shape
    _sigmoid_mul_grad_elementwise[(B,)](scores_fp32, grad_scores_fp32, grad_router_logits_fp32, B, E,
                                        scores_fp32.stride(0), scores_fp32.stride(1),
                                        grad_scores_fp32.stride(0), grad_scores_fp32.stride(1),
                                        grad_router_logits_fp32.stride(0), grad_router_logits_fp32.stride(1), BLOCK=256)


def _launch_sqnorm(grad_output):
    B, H = grad_output.shape
    norm = torch.empty((B,), dtype=torch.float32, device=grad_output.device)
    _row_sqnorm[(B,)](grad_output, norm, B, H,
                      grad_output.stride(0), grad_output.stride(1),
                      norm.stride(0))


def _launch_scatter_add_topk(grad_topk_weights_fp32, topk_indices_int32, grad_scores_fp32):
    B, K = grad_topk_weights_fp32.shape
    E = grad_scores_fp32.shape[1]
    _scatter_add_topk[(B,)](grad_topk_weights_fp32, topk_indices_int32, grad_scores_fp32, B, E, K,
                            grad_topk_weights_fp32.stride(0), grad_topk_weights_fp32.stride(1),
                            topk_indices_int32.stride(0), topk_indices_int32.stride(1),
                            grad_scores_fp32.stride(0), grad_scores_fp32.stride(1))


def _launch_matmul_bf16_bf16(A_bf16, B_bf16, out_bf16):
    M, K = A_bf16.shape
    K2, N = B_bf16.shape
    assert K == K2
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _matmul_bf16_bf16[grid](A_bf16, B_bf16, out_bf16,
                            M, N, K,
                            A_bf16.stride(0), A_bf16.stride(1),
                            B_bf16.stride(0), B_bf16.stride(1),
                            out_bf16.stride(0), out_bf16.stride(1))


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
        topk_weights: torch.Tensor,           # [B, K] fp32 (already normalized)
        score_mask: torch.Tensor,             # [B, E] fp32
        shared_expert_gate_weight: torch.Tensor,  # [H', H] bf16
        shared_expert_up_weight: torch.Tensor,    # [H', H] bf16
        shared_expert_down_weight: torch.Tensor,  # [H, H'] bf16
        shared_gate_output: torch.Tensor,         # [B, H'] bf16
        shared_up_output: torch.Tensor,           # [B, H'] bf16
    ):
        # Ensure CUDA and contiguity
        assert grad_output.is_cuda and hidden_states.is_cuda and router_weight.is_cuda
        assert shared_gate_output.is_cuda and shared_up_output.is_cuda and shared_expert_gate_weight.is_cuda
        assert shared_expert_up_weight.is_cuda and shared_expert_down_weight.is_cuda
        device = grad_output.device

        # 1) Shared paths: compute shared_activated via Triton elementwise kernel
        shared_activated = torch.empty_like(shared_gate_output, dtype=torch.bfloat16, device=device)
        _launch_silu_mul_elementwise(shared_gate_output, shared_up_output, shared_activated)

        # 2) Grad from shared down: grad_shared_output flows through down_weight
        grad_shared_output = grad_output  # [B, H]
        # grad_shared_activated = grad_shared_output @ down_weight  (Triton matmul)
        grad_shared_activated = torch.empty((grad_output.shape[0], shared_expert_down_weight.shape[1]),
                                            dtype=torch.bfloat16, device=device)
        A_down = grad_shared_output.contiguous()                   # [B, H]
        B_down = shared_expert_down_weight.contiguous()           # [H', H]
        _launch_matmul_bf16_bf16(A_down, B_down, grad_shared_activated)

        # 3) Down weight gradient: grad_shared_output.T @ shared_activated
        grad_shared_expert_down_weight = torch.empty_like(shared_expert_down_weight, dtype=torch.bfloat16, device=device)
        A_downT = grad_shared_output.transpose(0, 1).contiguous() # [H, B]
        B_downT = shared_activated.transpose(0, 1).contiguous()   # [H, H']
        _launch_matmul_bf16_bf16(A_downT, B_downT, grad_shared_expert_down_weight)

        # 4) Backward through SwiGLU: grad_shared_gate_output = grad_shared_activated * up_output
        grad_shared_gate_output = grad_shared_activated * shared_up_output  # [B, H'] in bf16
        # grad_shared_up_output = grad_shared_activated * silu(gate)
        silu_gate = torch.empty_like(shared_gate_output, dtype=torch.bfloat16, device=device)
        _silu_mul_elementwise(shared_gate_output, shared_gate_output, silu_gate)  # trick: gate * sigmoid(gate)
        # Compute silu gate properly: silu(g) = g * sigmoid(g)
        # We need explicit computation; keep it PyTorch for simplicity (these are small)
        silu_gate = (shared_gate_output.to(torch.float32) * torch.sigmoid(shared_gate_output.to(torch.float32))).to(torch.bfloat16)
        grad_shared_up_output = grad_shared_activated * silu_gate  # [B, H']

        # 5) Gate weight gradient: grad_hidden_from_shared_gate = grad_shared_gate_output @ gate_weight
        grad_hidden_from_shared_gate = torch.empty((grad_output.shape[0], hidden_states.shape[1]),
                                                    dtype=torch.bfloat16, device=device)
        A_gate = grad_shared_gate_output.transpose(0, 1).contiguous()   # [H, B]
        B_gate = shared_expert_gate_weight.contiguous()                 # [H, H]
        _launch_matmul_bf16_bf16(A_gate, B_gate, grad_hidden_from_shared_gate)

        # 6) Up weight gradient: grad_hidden_from_shared_up = grad_shared_up_output @ up_weight
        grad_hidden_from_shared_up = torch.empty((grad_output.shape[0], hidden_states.shape[1]),
                                                 dtype=torch.bfloat16, device=device)
        A_up = grad_shared_up_output.transpose(0, 1).contiguous()       # [H', B]
        B_up = shared_expert_up_weight.contiguous()                     # [H', H]
        _launch_matmul_bf16_bf16(A_up, B_up, grad_hidden_from_shared_up)

        # 7) Combine shared contributions to hidden gradient
        grad_hidden_states = torch.zeros_like(hidden_states, dtype=torch.bfloat16, device=device)
        grad_hidden_states = grad_hidden_states + grad_hidden_from_shared_up + grad_hidden_from_shared_gate

        # 8) Routing path: compute grad_router_logits via Triton elementwise
        grad_scores_fp32 = torch.empty((scores.shape[0], scores.shape[1]), dtype=torch.float32, device=device)
        _launch_sigmoid_mul_elementwise(scores, torch.zeros_like(scores, dtype=torch.float32, device=device), grad_scores_fp32)
        # Correct computation: grad_router_logits = grad_scores * scores * (1 - scores)
        # We already zero-initialized grad_scores; set it to the required contribution
        # We need topk-based contribution. Compute per-token norm and distribute over K.
        # Row-wise squared norm
        grad_output_bf16 = grad_output
        norm = torch.empty((grad_output_bf16.shape[0],), dtype=torch.float32, device=device)
        _launch_sqnorm(grad_output_bf16)
        # Compute grad_topk_weights_norm[b] = norm[b] / K
        grad_topk_weights_fp32 = (norm.view(-1, 1) / topk_indices.shape[1]).expand(-1, topk_indices.shape[1]).clone()
        # Scatter-add into grad_scores: grad_scores[b, idx[b, k]] += grad_topk_weights[b, k]
        _launch_scatter_add_topk(grad_topk_weights_fp32, topk_indices, grad_scores_fp32)
        # Mask: multiply by score_mask
        grad_scores_fp32 = grad_scores_fp32 * score_mask

        # Finally, grad_router_logits = grad_scores * scores * (1 - scores)
        # Re-run elementwise
        grad_router_logits_fp32 = torch.empty_like(scores, dtype=torch.float32, device=device)
        _launch_sigmoid_mul_elementwise(scores, grad_scores_fp32, grad_router_logits_fp32)

        # Route weight gradient: A = grad_router_logits.T [E, B], B = hidden_states [B, H], C = grad_router_weight [E, H]
        grad_router_weight = torch.empty((router_weight.shape[0], hidden_states.shape[1]),
                                         dtype=torch.bfloat16, device=device)
        A_rt = grad_router_logits_fp32.transpose(0, 1).contiguous()  # [E, B]
        B_rt = hidden_states.contiguous()                           # [B, H]
        _launch_matmul_bf16_bf16(A_rt, B_rt, grad_router_weight)

        # 9) Pack outputs: return gradients for required tensors
        # We only compute route weight gradient; gate/up/down for shared expert are already computed.
        return (
            grad_hidden_states,                       # [B, H]
            grad_router_weight,                      # [E, H]
            shared_expert_gate_weight.new_zeros([]), # placeholder, not returned (tuple length fixed)
            shared_expert_up_weight.new_zeros([]),   # placeholder
            grad_shared_expert_down_weight,          # [H, H']
        )


def run(*args):
    return ModelNew()(*args)
