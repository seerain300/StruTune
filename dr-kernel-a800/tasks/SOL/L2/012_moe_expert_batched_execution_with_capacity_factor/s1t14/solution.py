import torch
import triton
import triton.language as tl


@triton.jit
def bmm_triton_forward(A_ptr, B_ptr, Y_ptr,
                        M, K, N,
                        A_stride_m, A_stride_k,
                        B_stride_k, B_stride_n,
                        Y_stride_m, Y_stride_n,
                        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    # Each program computes a block of columns of Y along N
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator for [M, N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K (reduction) dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: [M, BLOCK_K]
        # M is usually 1 here (per-token vector), but keep generality
        a_ptrs = A_ptr + 0 * A_stride_m + offs_k[None, :] * A_stride_k  # rows are 0..M-1
        a_mask = (offs_k[None, :] < K) & (0 < M)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile: [BLOCK_K, N]
        b_ptrs = B_ptr + offs_k[:, None] * B_stride_k + offs_n[None, :] * B_stride_n
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        # a is [BLOCK_M, BLOCK_K], b is [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, b)

    # Store Y: [M, N]
    y_ptrs = Y_ptr + 0 * Y_stride_m + offs_n[None, :] * Y_stride_n
    y_mask = (0 < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def silu_mul_triton(Z_ptr, U_ptr, Y_ptr, L,
                    BLOCK_L: tl.constexpr):
    # Elementwise: Y[i] = silu(Z[i]) * U[i], i in [0, L)
    pid = tl.program_id(0)
    offs = pid * BLOCK_L + tl.arange(0, BLOCK_L)
    mask = offs < L

    z = tl.load(Z_ptr + offs, mask=mask, other=0.0)
    u = tl.load(U_ptr + offs, mask=mask, other=0.0)
    # silu(x) = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-z))
    y = (z * sig) * u
    tl.store(Y_ptr + offs, y, mask=mask)


@triton.jit
def atomic_accumulate_triton(W_ptr, X_ptr, Out_ptr, T,
                             M, N,
                             Out_stride_t, Out_stride_m):
    # Each program handles one token t and a block of columns m in [0, N)
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs_m < N

    # Load scalar weight w = W[t]
    w = tl.load(W_ptr + T)
    # Load X vector block x = X[t, offs_m]
    x_ptrs = X_ptr + T * M + offs_m
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    # Atomic add to Out[t, offs_m] += w * x
    out_ptrs = Out_ptr + T * Out_stride_t + offs_m * Out_stride_m
    tl.atomic_add(out_ptrs, w * x, mask=mask)


def _triton_bmm_forward(A, B, out, BLOCK_M=1, BLOCK_K=128, BLOCK_N=128, num_warps=4):
    """
    A: [M, K], B: [K, N], out: [M, N]
    All inputs/outputs are fp32 tensors. A and B must be contiguous; out is allocated.
    """
    assert A.is_cuda and B.is_cuda and out.is_cuda, "Triton bmm requires CUDA tensors"
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, "Inner dimensions must match"
    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    # Strides
    A_stride_m, A_stride_k = A.stride()
    B_stride_k, B_stride_n = B.stride()
    # out strides
    out_stride_m, out_stride_n = out.stride()
    grid = (triton.cdiv(N, BLOCK_N),)
    bmm_triton_forward[grid](
        A, B, out,
        M, K, N,
        A_stride_m, A_stride_k,
        B_stride_k, B_stride_n,
        out_stride_m, out_stride_n,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )


def _triton_silu_mul(Z, U, Y, BLOCK_L=1024, num_warps=4):
    """
    Elementwise: Y = silu(Z) * U.
    Z, U, Y must be 1D tensors of length L on CUDA, dtype fp32.
    """
    assert Z.is_cuda and U.is_cuda and Y.is_cuda, "Triton elementwise requires CUDA tensors"
    L = Z.numel()
    grid = (triton.cdiv(L, BLOCK_L),)
    silu_mul_triton[grid](Z, U, Y, L, BLOCK_L=BLOCK_L, num_warps=num_warps)


def _triton_atomic_accumulate(W, X, Out, T, BLOCK_M=128, num_warps=4):
    """
    Atomically accumulate Out[t, :] += W[t] * X[t, :].
    W: [1] fp32 scalar per token (we pass routing_weights[t, e] as a 1-element fp32 tensor).
    X: [1, N] fp32.
    Out: [num_tokens, hidden_size] fp32.
    """
    assert W.is_cuda and X.is_cuda and Out.is_cuda, "Triton atomic requires CUDA tensors"
    M, N = X.shape  # X is [1, N], but we pass M,N accordingly
    grid = (triton.cdiv(N, BLOCK_M),)
    atomic_accumulate_triton[grid](W, X, Out, T, M, N, Out.stride(0), Out.stride(1), BLOCK_M=BLOCK_M, num_warps=num_warps)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_gate_weights: torch.Tensor,
        expert_up_weights: torch.Tensor,
        expert_down_weights: torch.Tensor,
    ):
        """
        hidden_states: [num_tokens, hidden_size], bfloat16, CUDA
        selected_experts: [num_tokens, num_experts_per_tok], int64, CUDA
        routing_weights: [num_tokens, num_experts_per_tok], bfloat16, CUDA
        expert_gate_weights: [num_experts, hidden_size, moe_intermediate_size], bfloat16, CUDA
        expert_up_weights: [num_experts, hidden_size, moe_intermediate_size], bfloat16, CUDA
        expert_down_weights: [num_experts, moe_intermediate_size, hidden_size], bfloat16, CUDA
        Returns: [num_tokens, hidden_size], bfloat16
        """
        assert hidden_states.is_cuda and selected_experts.is_cuda and routing_weights.is_cuda, "All inputs must be on CUDA"
        assert expert_gate_weights.is_cuda and expert_up_weights.is_cuda and expert_down_weights.is_cuda, "All expert weights must be on CUDA"

        num_tokens, hidden_size = hidden_states.shape
        num_experts, H_in, H_out = expert_gate_weights.shape  # H_in == hidden_size, H_out == moe_intermediate_size
        num_experts_per_tok = selected_experts.shape[1]

        # Accumulator in fp32
        result = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=hidden_states.device)

        # Iterate tokens and selected experts; no torch ops on tensors
        for t in range(num_tokens):
            # For each expert j selected for token t
            for j in range(num_experts_per_tok):
                expert_id = int(selected_experts[t, j].item())
                # hidden_vec is [hidden_size] (bfloat16). We'll pass it to bmm as fp32
                hidden_vec = hidden_states[t].contiguous()  # [hidden_size], bfloat16
                hidden_vec_fp32 = hidden_vec.to(torch.float32)

                # 1) gate_out = hidden_vec @ expert_gate_weights[expert_id] -> [H_out], fp32
                gate_out = torch.empty(1, H_out, dtype=torch.float32, device=hidden_states.device)
                _triton_bmm_forward(hidden_vec_fp32[None], expert_gate_weights[expert_id], gate_out)

                # 2) up_out = hidden_vec @ expert_up_weights[expert_id] -> [H_out], fp32
                up_out = torch.empty(1, H_out, dtype=torch.float32, device=hidden_states.device)
                _triton_bmm_forward(hidden_vec_fp32[None], expert_up_weights[expert_id], up_out)

                # 3) activated = silu(gate_out) * up_out
                activated = torch.empty(1, H_out, dtype=torch.float32, device=hidden_states.device)
                _triton_silu_mul(gate_out, up_out, activated)

                # 4) final_out = activated @ expert_down_weights[expert_id] -> [hidden_size], fp32
                final_out = torch.empty(1, hidden_size, dtype=torch.float32, device=hidden_states.device)
                # activated is [1, H_out], expert_down_weights[expert_id] is [H_out, hidden_size]
                _triton_bmm_forward(activated, expert_down_weights[expert_id], final_out)

                # 5) Accumulate: result[t] += routing_weights[t, j] * final_out
                # routing_weights[t, j] is bfloat16; convert to fp32 scalar tensor for kernel
                weight = routing_weights[t, j].to(torch.float32).unsqueeze(0)  # [1], fp32
                _triton_atomic_accumulate(weight, final_out, result, t)

        # Return fp32 result; the caller can cast to bfloat16 if needed
        return result


def run(*args):
    return ModelNew()(*args)
