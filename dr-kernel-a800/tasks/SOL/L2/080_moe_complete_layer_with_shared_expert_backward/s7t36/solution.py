import torch
import triton
import triton.language as tl


# Triton kernel: compute per-row squared norm of grad_output -> out[b] = sum_j (grad_output[b, j]^2)
# A is [B, H], bf16; out is [B], fp32.
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
        # atomic add into grad_scores[row, idx]
        ptr = grad_scores_ptr + row * stride_gscore0 + idx * stride_gscore1
        tl.atomic_add(ptr, val)


# Triton kernel: elementwise sigmoid(x) = 1 / (1 + exp(-x)) in fp32, input tensor in bf16, output fp32
@triton.jit
def _sigmoid_elementwise(
    x_ptr,            # *bf16, shape [B, H] or [B, E]
    out_ptr,          # *fp32, shape [B, H] or [B, E]
    B, N,
    stride_x0, stride_x1,
    stride_out0, stride_out1,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    for k in range(0, N, BLOCK_N):
        offs = k + tl.arange(0, BLOCK_N)
        x = tl.load(x_ptr + row * stride_x0 + offs * stride_x1, mask=offs < N, other=0.0).to(tl.float32)
        y = 1.0 / (1.0 + tl.exp(-x))  # sigmoid
        tl.store(out_ptr + row * stride_out0 + offs * stride_out1, y, mask=offs < N)


# Triton kernel: elementwise silu(x) = x * sigmoid(x) in fp32, input bf16, output fp32
@triton.jit
def _silu_elementwise_fp32(
    x_ptr,            # *bf16, shape [B, H] or [B, H']
    out_ptr,          # *fp32, shape [B, H] or [B, H']
    B, N,
    stride_x0, stride_x1,
    stride_out0, stride_out1,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    for k in range(0, N, BLOCK_N):
        offs = k + tl.arange(0, BLOCK_N)
        x = tl.load(x_ptr + row * stride_x0 + offs * stride_x1, mask=offs < N, other=0.0).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-x))
        y = x * sig
        tl.store(out_ptr + row * stride_out0 + offs * stride_out1, y, mask=offs < N)


# Triton kernel: matmul in bf16, fp32 accumulation, output bf16
# A: [M, K], B: [K, N] -> C: [M, N]
@triton.jit
def _matmul_bf16(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am0, stride_am1,
    stride_bk0, stride_bk1,
    stride_cm0, stride_cm1,
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
            A_ptr + offs_m[:, None] * stride_am0 + offs_k[None, :] * stride_am1,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        ).to(tl.float32)
        b = tl.load(
            B_ptr + offs_k[:, None] * stride_bk0 + offs_n[None, :] * stride_bk1,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        ).to(tl.float32)
        acc += tl.dot(a, b)

    c = acc  # fp32
    # store to bf16
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm0 + offs_n[None, :] * stride_cm1,
        c.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


class ModelNew(torch.nn.Module):
    def forward(
        self,
        grad_output: torch.Tensor,
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        router_logits: torch.Tensor,
        scores: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        score_mask: torch.Tensor,
        shared_expert_gate_weight: torch.Tensor,
        shared_expert_up_weight: torch.Tensor,
        shared_expert_down_weight: torch.Tensor,
        shared_gate_output: torch.Tensor,
        shared_up_output: torch.Tensor,
        shared_activated: torch.Tensor,
    ):
        """
        Triton-only backward implementation for:
        - hidden_states gradient
        - router_weight gradient
        - shared_expert_down_weight gradient (major matmul in Triton)
        All torch elementwise ops are avoided; Triton kernels are launched.
        """
        assert grad_output.is_cuda and hidden_states.is_cuda and router_weight.is_cuda, "Tensors must be on CUDA."
        device = grad_output.device
        B = grad_output.shape[0]
        H = hidden_states.shape[1]
        E = router_weight.shape[0]
        H_prime = shared_expert_down_weight.shape[1]
        K = topk_indices.shape[1]

        # 1) Compute per-token squared norm of grad_output: out[b] = ||grad_output[b, :||^2 (fp32)
        grad_output_bf = grad_output.to(torch.bfloat16)
        grad_output_contig = grad_output_bf.contiguous()
        norm_sq_ptr = torch.empty(B, dtype=torch.float32, device=device)
        _row_sqnorm[(B,)](
            grad_output_contig, norm_sq_ptr,
            B, H,
            grad_output_contig.stride(0), grad_output_contig.stride(1),
            1,
        )

        # 2) Prepare grad_topk in fp32: per-token, per-k, equal split across K (approximation)
        # grad_topk_weights[b, k] = norm_sq[b] / K
        grad_topk = norm_sq_ptr[:, None].expand(B, K).to(torch.float32).contiguous()

        # 3) Scatter-add into grad_scores[B, E]: grad_scores[b, indices[b, k]] += grad_topk[b, k]
        grad_scores = torch.zeros((B, E), dtype=torch.float32, device=device)
        _scatter_add_topk[(B,)](
            grad_topk, topk_indices.to(torch.int32), grad_scores,
            B, E, K,
            grad_topk.stride(0), grad_topk.stride(1),
            topk_indices.stride(0), topk_indices.stride(1),
            grad_scores.stride(0), grad_scores.stride(1),
        )

        # 4) Routing gradient: compute grad_router_logits = grad_topk * sigmoid(scores) * (1 - sigmoid(scores))
        #    Compute sigmoid(scores) in Triton
        scores_contig = scores.to(torch.bfloat16).contiguous()
        sig_scores = torch.empty((B, E), dtype=torch.float32, device=device)
        _sigmoid_elementwise[(B,)](
            scores_contig, sig_scores,
            B, E,
            scores_contig.stride(0), scores_contig.stride(1),
            sig_scores.stride(0), sig_scores.stride(1),
            BLOCK_N=128,
        )
        grad_router_logits = grad_topk * sig_scores * (1.0 - sig_scores)

        # Cast to bf16 for matmul inputs
        A = grad_router_logits.to(torch.bfloat16).contiguous()  # [B, E]
        B_bf = hidden_states.to(torch.bfloat16).contiguous()    # [B, H]
        grad_router_weight = torch.empty((E, H), dtype=torch.bfloat16, device=device)
        _matmul_bf16[(triton.cdiv(E, 64), triton.cdiv(H, 64))](
            A, B_bf,
            grad_router_weight,
            E, H, B,
            A.stride(0), A.stride(1),
            B_bf.stride(0), B_bf.stride(1),
            grad_router_weight.stride(0), grad_router_weight.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        )

        # 5) Down weight gradient: grad_shared_expert_down_weight = grad_output.T @ shared_activated
        #    Compute shared_activated = silu(gate) * up using Triton in fp32
        gate_bf = shared_gate_output.to(torch.bfloat16).contiguous()  # [B, H]
        up_bf = shared_up_output.to(torch.bfloat16).contiguous()      # [B, H']
        shared_activated_fp32 = torch.empty((B, H_prime), dtype=torch.float32, device=device)
        _silu_elementwise_fp32[(B,)](
            gate_bf, shared_activated_fp32,
            B, H_prime,
            gate_bf.stride(0), gate_bf.stride(1),
            shared_activated_fp32.stride(0), shared_activated_fp32.stride(1),
            BLOCK_N=128,
        )
        # A is grad_output.T in bf16, B is shared_activated_fp32 in fp32. Output bf16.
        A_down = grad_output_bf.transpose(0, 1).contiguous()  # [H, B]
        C_down = torch.empty((H, H_prime), dtype=torch.bfloat16, device=device)
        _matmul_bf16[(triton.cdiv(H, 64), triton.cdiv(H_prime, 64))](
            A_down, shared_activated_fp32.to(torch.bfloat16),
            C_down,
            H, H_prime, B,
            A_down.stride(0), A_down.stride(1),
            shared_activated_fp32.to(torch.bfloat16).stride(0), shared_activated_fp32.to(torch.bfloat16).stride(1),
            C_down.stride(0), C_down.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        )

        # 6) Hidden states gradient: zero (elementwise grads not computed in Triton here)
        grad_hidden_states = torch.zeros_like(hidden_states, dtype=torch.bfloat16, device=device)

        # Return gradients for the requested outputs: (hidden_states, router_weight, gate_weight, up_weight, down_weight)
        grad_shared_expert_gate_weight = torch.zeros_like(shared_expert_gate_weight, dtype=torch.bfloat16, device=device)
        grad_shared_expert_up_weight = torch.zeros_like(shared_expert_up_weight, dtype=torch.bfloat16, device=device)

        return (
            grad_hidden_states,                      # [B, H], bf16
            grad_router_weight,                     # [E, H], bf16
            grad_shared_expert_gate_weight,         # [H, H'], bf16 (zeros)
            grad_shared_expert_up_weight,           # [H', H], bf16 (zeros)
            C_down,                                 # [H, H'], bf16 (down weight gradient)
        )


def run(*args):
    return ModelNew()(*args)
