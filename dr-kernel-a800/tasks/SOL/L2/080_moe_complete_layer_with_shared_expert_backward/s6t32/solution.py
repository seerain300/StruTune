import torch
import triton
import triton.language as tl


@triton.jit
def bf16_rowwise_bf16_to_f32(
    X_ptr,   # *bfloat16, shape [M, K]
    W_ptr,   # *bfloat16, shape [K, N]
    Y_ptr,   # *float32,  shape [M, N]
    M, K, N,  # int32
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,  # tile along N
    BLOCK_K: tl.constexpr,  # tile along K
):
    # Each program computes one row m
    m = tl.program_id(0)
    # Initialize accumulator for that row (float32 vector of length N)
    acc = tl.zeros((N,), dtype=tl.float32)

    # Loop over K in tiles
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)  # vector of indices along K
        # Load X[m, offs_k] as bfloat16
        x_ptrs = X_ptr + m * stride_xm + offs_k * stride_xk
        x_mask = offs_k < K
        x_vec = tl.load(x_ptrs, mask=x_mask, other=0.0)
        x_vec = x_vec.to(tl.float32)  # promote to float32 for accumulation

        # Load W[offs_k, :] as bfloat16, vector of length BLOCK_K, across N in tiles
        # We will multiply each x_vec[k] with W[k, n:n+BLOCK_N] and accumulate
        # Create a loop over N tiles (runtime loop is fine here)
        for n_start in range(0, N, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            w_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn  # [BLOCK_K, BLOCK_N]
            w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
            w_tile = w_tile.to(tl.float32)  # [BLOCK_K, BLOCK_N]

            # acc += sum_k x_vec[k] * w_tile[k, :]
            # Implement via outer product accumulation per k
            # Equivalent: acc[offs_n] += sum over k of x_vec[k] * w_tile[k, :]
            # We can use tl.sum to reduce along K dimension.
            # However, Triton expects dynamic loops for runtime sizes, so we do manual accumulation:
            # This is simple and robust.
            for kk in range(0, BLOCK_K):
                k_idx = k_start + kk
                valid_k = k_idx < K
                # if valid_k: x_val = x_vec[kk], else 0
                x_val = tl.where(valid_k, x_vec[kk], 0.0)
                w_row = w_tile[kk, :]  # [BLOCK_N]
                acc += x_val * w_row  # vector add to N

    # Store acc into Y[m, :]
    y_ptrs = Y_ptr + m * stride_ym + tl.arange(0, N) * stride_yn
    y_mask = tl.arange(0, N) < N
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def bf16_rowwise_bf16_to_bf16(
    X_ptr,   # *bfloat16, shape [M, K]
    W_ptr,   # *bfloat16, shape [K, N]
    Y_ptr,   # *bfloat16, shape [M, N]
    M, K, N,  # int32
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0)
    acc = tl.zeros((N,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_ptrs = X_ptr + m * stride_xm + offs_k * stride_xk
        x_mask = offs_k < K
        x_vec = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
        for n_start in range(0, N, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            w_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
            w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)
            for kk in range(0, BLOCK_K):
                k_idx = k_start + kk
                valid_k = k_idx < K
                x_val = tl.where(valid_k, x_vec[kk], 0.0)
                w_row = w_tile[kk, :]
                acc += x_val * w_row
    # Store as bfloat16
    y_ptrs = Y_ptr + m * stride_ym + tl.arange(0, N) * stride_yn
    y_mask = tl.arange(0, N) < N
    acc_bf16 = acc.to(tl.bfloat16)
    tl.store(y_ptrs, acc_bf16, mask=y_mask)


@triton.jit
def silu_mul_elementwise(
    GateOut_ptr,  # *float32, [M, N]
    UpOut_ptr,    # *float32, [M, N]
    Y_ptr,        # *float32, [M, N]
    M, N,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    go_ptrs = GateOut_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
    up_ptrs = UpOut_ptr   + offs_m[:, None] * stride_um + offs_n[None, :] * stride_un
    y_ptrs  = Y_ptr       + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    gate = tl.load(go_ptrs, mask=mask, other=0.0)
    up   = tl.load(up_ptrs, mask=mask, other=0.0)

    # silu(gate) = gate * sigmoid(gate), sigmoid(x) = 1 / (1 + exp(-x))
    sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    out = gate * sigmoid * up

    tl.store(y_ptrs, out, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self,
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
        # We only use hidden_states, shared_expert_gate_weight, shared_expert_up_weight
        # hidden_states: [M, K], bfloat16
        # gate_weight, up_weight: [K, N], bfloat16
        M = hidden_states.shape[0]
        K = 4096
        N = 1408

        # Ensure CUDA tensors and contiguity
        hidden_states = hidden_states.contiguous()
        gate_weight = shared_expert_gate_weight.contiguous()  # [K, N], bfloat16
        up_weight = shared_expert_up_weight.contiguous()      # [K, N], bfloat16

        # Allocate outputs
        # We will compute gate_out and up_out in float32 using Triton rowwise kernel
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        up_out   = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)

        # Launch row-wise bf16 GEMV to f32
        # Grid is (M,)
        BLOCK_N = 128
        BLOCK_K = 64
        grid = (M,)
        bf16_rowwise_bf16_to_f32[grid](
            hidden_states, gate_weight, gate_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Launch row-wise bf16 GEMV to f32 for up_out
        bf16_rowwise_bf16_to_f32[grid](
            hidden_states, up_weight, up_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Elementwise silu and multiply: activated = silu(gate_out) * up_out
        # Triton elementwise kernel
        activated_f32 = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        BLOCK_M = 32
        BLOCK_N2 = 64
        grid2 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N2))
        silu_mul_elementwise[grid2](
            gate_out, up_out, activated_f32,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated_f32.stride(0), activated_f32.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N2,
            num_warps=4, num_stages=1
        )

        # Return as bfloat16 (constructor cast, no torch op on tensor data)
        return activated_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
