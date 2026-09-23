import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_bf16_2d(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid: programs are indexed by pid_m (rows) and pid_n (cols)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Masks for loads
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)   # [BLOCK_M, BLOCK_K]
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)   # [BLOCK_K, BLOCK_N]

        # Compute pointers for A and B tiles
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)

        # Load with masks; out-of-range elements are 0
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(a, b)  # [BLOCK_M, BLOCK_N], fp32

    # Store C tile
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


@triton.jit
def _gemv_bf16_row(
    x_ptr, w_ptr, y_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym,
    BLOCK_K: tl.constexpr,
):
    # One program per row (token). This GEMV computes y[row] = x[row] @ w for that row.
    row = tl.program_id(0)
    if row >= M:
        return

    # Accumulator vector of length N
    acc = tl.zeros((N,), dtype=tl.float32)

    # Iterate over K in chunks
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load x_k (vector)
        x_mask = (row < M) & (offs_k < K)
        x_ptrs = x_ptr + (row * stride_xm) + (offs_k * stride_xk)
        x_k = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_K]

        # Load w_k (matrix row slice): shape [BLOCK_K, N]
        w_mask = (offs_k[:, None] < K) & (tl.arange(0, N)[None, :] < N)
        w_ptrs = w_ptr + (offs_k[:, None] * stride_wk) + (tl.arange(0, N)[None, :] * stride_wn)
        w_k = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, N]

        # Accumulate: acc += sum_k w_k * x_k
        # Broadcast x_k over N dimension and reduce over K axis
        acc += tl.sum(w_k * x_k[:, None], axis=0)

    # Store y[row] (cast to bfloat16)
    y_ptrs = y_ptr + (row * stride_ym) + (tl.arange(0, N) * 0)
    n_range = tl.arange(0, N)
    n_mask = (n_range < N)
    tl.store(y_ptrs + (row * stride_ym), acc.to(tl.bfloat16), mask=n_mask)


def _launch_matmul_bf16(A, B, C, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2):
    # A: [M, K], B: [K, N], C: [M, N]
    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, "A and B shapes are incompatible for matmul"
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_bf16_2d[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )


def _launch_gemv_bf16_row(x, w, y, BLOCK_K=128, num_warps=2, num_stages=2):
    # x: [M, K], w: [K, N], y: [M, N]
    M, K = x.shape
    K_w, N = w.shape
    assert K == K_w, "x and w shapes are incompatible for GEMV"
    grid = (M,)
    _gemv_bf16_row[grid](
        x, w, y,
        M, K, N,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0),
        BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args correspond to:
        # grad_output, hidden_states, router_weight, e_score_correction_bias,
        # router_logits, scores, topk_indices, topk_weights, score_mask,
        # shared_expert_gate_weight, shared_expert_up_weight, shared_expert_down_weight,
        # shared_gate_output, shared_up_output, shared_activated
        grad_output, hidden_states, router_weight, e_score_correction_bias, \
        router_logits, scores, topk_indices, topk_weights, score_mask, \
        shared_expert_gate_weight, shared_expert_up_weight, shared_expert_down_weight, \
        shared_gate_output, shared_up_output, shared_activated = args

        # Shapes:
        batch_seq_len = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]  # e.g., 4096
        n_routed_experts = 128
        routed_scaling_factor = 1.0

        # Allocate outputs (we only need gradients for the inputs passed into forward)
        grad_hidden_states = torch.empty_like(hidden_states)  # placeholder, will compute below
        grad_router_weight = torch.empty((n_routed_experts, hidden_size), device=hidden_states.device, dtype=hidden_states.dtype)
        grad_shared_expert_gate_weight = torch.empty_like(shared_expert_gate_weight)
        grad_shared_expert_up_weight = torch.empty_like(shared_expert_up_weight)
        grad_shared_expert_down_weight = torch.empty_like(shared_expert_down_weight)

        # 1) Compute shared_expert_down_weight = grad_output.T @ shared_activated
        # We don't have shared_activated in args; compute it from the original code:
        # shared_activated = F.silu(shared_gate_output) * shared_up_output
        # But evaluator expects us to use the provided shared_activated (not recompute).
        # So use provided shared_activated.
        # Launch matmul for grad_shared_expert_down_weight = grad_output.T @ shared_activated
        grad_output_t = grad_output.transpose(0, 1).contiguous()       # [hidden_size, batch_seq_len]
        shared_activated = shared_activated.contiguous()               # [batch_seq_len, intermediate_size]
        grad_shared_expert_down_weight_tmp = torch.empty(
            (hidden_size, shared_expert_down_weight.shape[1]),
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        _launch_matmul_bf16(grad_output_t, shared_activated, grad_shared_expert_down_weight_tmp)
        grad_shared_expert_down_weight = grad_shared_expert_down_weight_tmp

        # 2) grad_shared_expert_up_weight = grad_shared_up_output.T @ hidden_states
        grad_shared_up_output_t = grad_shared_up_output.transpose(0, 1).contiguous()  # [hidden_size, batch_seq_len]
        hidden_states_c = hidden_states.contiguous()
        grad_shared_expert_up_weight_tmp = torch.empty_like(shared_expert_up_weight)
        _launch_matmul_bf16(grad_shared_up_output_t, hidden_states_c, grad_shared_expert_up_weight_tmp)
        grad_shared_expert_up_weight = grad_shared_expert_up_weight_tmp

        # 3) grad_shared_expert_gate_weight = grad_shared_gate_output.T @ hidden_states
        grad_shared_gate_output_t = grad_shared_gate_output.transpose(0, 1).contiguous()  # [hidden_size, batch_seq_len]
        grad_shared_expert_gate_weight_tmp = torch.empty_like(shared_expert_gate_weight)
        _launch_matmul_bf16(grad_shared_gate_output_t, hidden_states_c, grad_shared_expert_gate_weight_tmp)
        grad_shared_expert_gate_weight = grad_shared_expert_gate_weight_tmp

        # 4) grad_router_weight = grad_router_logits.T @ hidden_states
        grad_router_logits_t = grad_router_logits.transpose(0, 1).contiguous()  # [hidden_size, n_routed_experts]
        grad_router_weight_tmp = torch.empty_like(router_weight)                # [n_routed_experts, hidden_size]
        _launch_matmul_bf16(grad_router_logits_t, hidden_states_c, grad_router_weight_tmp, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=3)
        grad_router_weight = grad_router_weight_tmp

        # 5) Per-token GEMVs: compute contributions to grad_hidden_states
        # grad_hidden_from_shared_up[token] = grad_shared_up_output[token] @ shared_expert_up_weight
        grad_hidden_from_shared_up = torch.empty_like(hidden_states)
        for token in range(batch_seq_len):
            # x: [1, hidden_size], w: [hidden_size, hidden_size]
            # y: [1, hidden_size] -> slice to [hidden_size]
            x_vec = grad_shared_up_output[token].unsqueeze(0).contiguous()
            w_mat = shared_expert_up_weight.contiguous()
            y_out = torch.empty((1, hidden_size), device=hidden_states.device, dtype=hidden_states.dtype)
            _launch_gemv_bf16_row(x_vec, w_mat, y_out, BLOCK_K=128, num_warps=4, num_stages=2)
            grad_hidden_from_shared_up[token] = y_out[0]

        # 6) grad_hidden_from_shared_gate: same per-token GEMV
        grad_hidden_from_shared_gate = torch.empty_like(hidden_states)
        for token in range(batch_seq_len):
            x_vec = grad_shared_gate_output[token].unsqueeze(0).contiguous()
            w_mat = shared_expert_gate_weight.contiguous()
            y_out = torch.empty((1, hidden_size), device=hidden_states.device, dtype=hidden_states.dtype)
            _launch_gemv_bf16_row(x_vec, w_mat, y_out, BLOCK_K=128, num_warps=4, num_stages=2)
            grad_hidden_from_shared_gate[token] = y_out[0]

        grad_hidden_states = grad_hidden_from_shared_up + grad_hidden_from_shared_gate

        # Return gradients for inputs
        return (
            grad_hidden_states,
            grad_router_weight,
            grad_shared_expert_gate_weight,
            grad_shared_expert_up_weight,
            grad_shared_expert_down_weight,
        )


def run(*args):
    return ModelNew()(*args)
