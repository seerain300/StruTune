import torch
import triton
import triton.language as tl


# Triton kernel: fill a 1D buffer with random normal values (fp32)
# This is used to generate hidden_states, grad_output, and router_weight.
@triton.jit
def _randn_fill_triton(OUT_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.rand(offsets.to(tl.int32), seed=0)  # device-side RNG; seed arbitrary
    tl.store(OUT_ptr + offsets, vals, mask=mask)  # fp32 randoms


# Triton elementwise sigmoid: y = 1 / (1 + exp(-x)) (fp32)
@triton.jit
def _sigmoid_triton(IN_ptr, OUT_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(IN_ptr + offsets, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(OUT_ptr + offsets, y, mask=mask)


# Triton elementwise softplus: y = log(1 + exp(x)) (fp32)
@triton.jit
def _softplus_triton(IN_ptr, OUT_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(IN_ptr + offsets, mask=mask, other=0.0)
    y = tl.log(1.0 + tl.exp(x))
    tl.store(OUT_ptr + offsets, y, mask=mask)


# Triton GEMM kernel: C[M, N] = A[M, K] @ B[N, K] where B is W.T with shape [N, K]
# Accumulate in fp32, write fp32 outputs.
@triton.jit
def _matmul_triton_fp32(
    A_ptr,   # *fp32, [M, K]
    B_ptr,   # *fp32, [N, K] (W.T)
    C_ptr,   # *fp32, [M, N]
    M, N, K,
    stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (k_ids[None, :] * stride_ak)
        B_ptrs = B_ptr + (offs_n[None, :] * stride_bn) + (k_ids[:, None] * stride_bk)
        a_mask = (offs_m[:, None] < M) & (k_ids[None, :] < K)
        b_mask = (offs_n[None, :] < N) & (k_ids[:, None] < K)
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)
        B_tile = tl.load(B_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(A_tile, B_tile)

    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=c_mask)


# Triton elementwise multiply: OUT = A * B (fp32)
@triton.jit
def _elementwise_mul(A_ptr, B_ptr, OUT_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(OUT_ptr + offsets, a * b, mask=mask)


# Triton kernel: per-row top-k (descending) for a 2D matrix IN[M, N]
# Writes OUT_idx[M, K] = indices (int32), OUT_w[M, K] = weights (fp32).
@triton.jit
def _topk_rows_kernel(IN_ptr, OUT_idx_ptr, OUT_w_ptr, M, N, K, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base_in = row * N
    top_vals = tl.full((K,), -1.0e30, dtype=tl.float32)
    top_idxs = tl.full((K,), -1, dtype=tl.int32)

    for j in range(0, N):
        val = tl.load(IN_ptr + base_in + j)
        # Insertion into top-k
        for kk in range(0, K):
            cond = val > top_vals[kk]
            new_val = tl.where(cond, val, top_vals[kk])
            new_idx = tl.where(cond, j, top_idxs[kk])
            # Shift down
            for r in range(K - 1, kk, -1):
                top_vals[r] = top_vals[r - 1]
                top_idxs[r] = top_idxs[r - 1]
            top_vals[kk] = new_val
            top_idxs[kk] = new_idx

    # Store results
    for kk in range(0, K):
        tl.store(OUT_idx_ptr + row * K + kk, top_idxs[kk])
        tl.store(OUT_w_ptr + row * K + kk, top_vals[kk])


# Triton kernel: normalize top-k weights per row by denom and scale
# OUT_w[M*K] in row-major: write normalized values back
@triton.jit
def _normalize_topk(OUT_w_ptr, DENOM_ptr, SCALE, M, K, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = rows < M
    denom = tl.load(DENOM_ptr + rows, mask=mask, other=1.0)
    eps = 1.0e-20
    base = rows * K
    offs = tl.arange(0, BLOCK_M)
    for j in range(0, K):
        ptrs = OUT_w_ptr + base + j
        w = tl.load(ptrs, mask=mask, other=0.0)
        w = w * SCALE / (denom + eps)
        tl.store(ptrs, w, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, batch_seq_len: int):
        # Avoid torch operations in forward; allocate with torch.empty if needed.
        device_index = torch.cuda.current_device()
        device = torch.device("cuda", device_index)

        # Constants to match original signature (these are dynamic per workload)
        H = 4096
        n_routed_experts = 128
        num_experts_per_tok = 8
        routed_scaling_factor = 1.0
        shared_expert_intermediate = 1408

        # 1) Generate grad_output and hidden_states (fp32 output buffers; will cast if needed)
        grad_output = torch.empty((batch_seq_len, H), dtype=torch.bfloat16, device=device)
        hidden_states = torch.empty((batch_seq_len, H), dtype=torch.bfloat16, device=device)

        total_hs = batch_seq_len * H
        grid_fill = (_cdiv(total_hs, 1024),)
        _randn_fill_triton[grid_fill](grad_output.view(-1), total_hs, BLOCK=1024)
        _randn_fill_triton[grid_fill](hidden_states.view(-1), total_hs, BLOCK=1024)

        # 2) Generate router_weight (bfloat16)
        router_weight = torch.empty((n_routed_experts, H), dtype=torch.bfloat16, device=device)
        _randn_fill_triton[(n_routed_experts * H,)](router_weight.view(-1), n_routed_experts * H, BLOCK=1024)

        # 3) Compute logits = hidden_states @ router_weight.T (fp32) via Triton
        # Create random A and W since we don't have real inputs in forward
        A = torch.empty((batch_seq_len, H), dtype=torch.float32, device=device)
        Wt = torch.empty((H, n_routed_experts), dtype=torch.float32, device=device)
        _randn_fill_triton[(batch_seq_len * H,)](A.view(-1), batch_seq_len * H, BLOCK=1024)
        _randn_fill_triton[(H * n_routed_experts,)](Wt.view(-1), H * n_routed_experts, BLOCK=1024)

        logits = torch.empty((batch_seq_len, n_routed_experts), dtype=torch.float32, device=device)
        grid_log = (triton.cdiv(batch_seq_len, 64), triton.cdiv(n_routed_experts, 64))
        _matmul_triton_fp32[grid_log](
            A, Wt, logits,
            batch_seq_len, n_routed_experts, H,
            A.stride(0), A.stride(1), Wt.stride(0), Wt.stride(1), logits.stride(0), logits.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, num_warps=4, num_stages=3
        )

        # 4) Compute scores = sigmoid(logits)
        scores = torch.empty((batch_seq_len, n_routed_experts), dtype=torch.float32, device=device)
        grid_sig = (_cdiv(batch_seq_len * n_routed_experts, 1024),)
        _sigmoid_triton[grid_sig](logits.view(-1), scores.view(-1), batch_seq_len * n_routed_experts, BLOCK=1024)

        # 5) Top-k selection: we need topk_indices and topk_weights (fp32)
        topk_indices = torch.empty((batch_seq_len, num_experts_per_tok), dtype=torch.int32, device=device)
        topk_weights = torch.empty((batch_seq_len, num_experts_per_tok), dtype=torch.float32, device=device)

        _topk_rows_kernel[(batch_seq_len,)](
            scores, topk_indices, topk_weights,
            batch_seq_len, n_routed_experts, num_experts_per_tok, BLOCK_N=128
        )

        # 6) Normalize top-k weights: denom = sum(topk_weights, dim=1), w_norm = w / denom * routed_scaling_factor
        denom = topk_weights.sum(dim=1, keepdim=True) + 1e-20  # [batch_seq_len, 1] fp32
        _normalize_topk[(triton.cdiv(batch_seq_len, 128),)](
            topk_weights, denom.view(-1), routed_scaling_factor,
            batch_seq_len, num_experts_per_tok, BLOCK_M=128
        )

        # 7) Compute shared_expert gate and up outputs (fp32) via Triton with random A/W
        A_hs = torch.empty((batch_seq_len, H), dtype=torch.float32, device=device)
        _randn_fill_triton[(batch_seq_len * H,)](A_hs.view(-1), batch_seq_len * H, BLOCK=1024)

        Wg = torch.empty((shared_expert_intermediate, H), dtype=torch.float32, device=device)  # gate weight


def run(*args):
    return ModelNew()(*args)
