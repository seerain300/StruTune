import torch
import triton
import triton.language as tl


# Triton kernel: fill 1D tensor with random normal values and cast to bfloat16
@triton.jit
def _randn_fill_triton(OUT_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Generate uniform random in [0, 1) then normal via Box-Muller
    u = tl.rand(offsets, mask=mask)
    v = tl.rand(offsets, mask=mask)
    x = tl.sqrt(-2.0 * tl.log(1.0 - u)) * tl.cos(2.0 * 3.141592653589793 * v)
    tl.store(OUT_ptr + offsets, x.to(tl.bfloat16), mask=mask)


# Triton matmul kernel: C[M, N] = A[M, K] @ B[N, K] where B is W.T (fp32 compute, fp32 output)
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


# Triton elementwise sigmoid: y = 1 / (1 + exp(-x))
@triton.jit
def _sigmoid_triton(in_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, y, mask=mask)


# Triton elementwise softplus: y = log(1 + exp(x)) (fp32 compute)
@triton.jit
def _softplus_triton(in_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    y = tl.log(1.0 + tl.exp(x))
    tl.store(out_ptr + offsets, y, mask=mask)


# Triton kernel: per-row top-k selection over a 2D [M, N] input (float32), returns topk_indices and topk_weights
# We implement a K-scan per row, selecting the maximum K elements with sorted=False. Indices are int32.
@triton.jit
def _topk_rows_kernel(IN_ptr, IND_ptr, WT_ptr, M, N, K: tl.constexpr):
    pid = tl.program_id(0)
    token = pid
    if token >= M:
        return
    row_base = token * N
    # Initialize topk values and indices
    top_vals = tl.full((K,), -1e30, dtype=tl.float32)
    top_inds = tl.full((K,), -1, dtype=tl.int32)

    # Scan across N columns and update top-k
    for j in range(0, N):
        val = tl.load(IN_ptr + row_base + j)
        # Try to insert into top-k positions
        for i in range(0, K):
            if val > top_vals[i]:
                tmp_val = val
                tmp_ind = j
                # Shift down from i+1 to 0
                for r in range(K - 1, i, -1):
                    top_vals[r] = top_vals[r - 1]
                    top_inds[r] = top_inds[r - 1]
                top_vals[i] = tmp_val
                top_inds[i] = tmp_ind
                break  # one insertion per outer loop

    # Write out results
    out_base = token * K
    for i in range(0, K):
        tl.store(IND_ptr + out_base + i, top_inds[i])
        tl.store(WT_ptr + out_base + i, top_vals[i])


# Triton kernel: per-token normalization of topk weights
# normalized = wt * (scale / (denom + eps)), where denom is sum of topk weights for that token and scale is routed_scaling_factor
@triton.jit
def _normalize_topk(IN_ptr, DENOM_ptr, OUT_ptr, M, K: tl.constexpr):
    token = tl.program_id(0)
    if token >= M:
        return
    base = token * K
    denom = tl.load(DENOM_ptr + token)  # scalar per token
    eps = 1e-20
    scale = 1.0  # routed_scaling_factor not provided by forward; default 1.0
    for i in range(0, K):
        v = tl.load(IN_ptr + base + i)
        tl.store(OUT_ptr + base + i, v * (scale / (denom + eps)))


# Triton elementwise multiply: OUT = A * B (fp32)
@triton.jit
def _elementwise_mul(A_ptr, B_ptr, OUT_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(OUT_ptr + offsets, a * b, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, batch_seq_len: int):
        # Select CUDA device without using torch operations in forward
        device_index = torch.cuda.current_device()
        device = torch.device("cuda", device_index)

        # Constants (to match original signature usage in workloads)
        H = 4096
        n_routed_experts = 128
        num_experts_per_tok = 8
        routed_scaling_factor = 1.0
        shared_expert_intermediate = 1408

        # 1) Generate required random tensors with Triton (bfloat16 output)
        grad_output = torch.empty((batch_seq_len, H), dtype=torch.bfloat16, device=device)
        hidden_states = torch.empty((batch_seq_len, H), dtype=torch.bfloat16, device=device)

        # Random fill for grad_output and hidden_states
        total_hs = batch_seq_len * H
        grid_fill = (_cdiv(total_hs, 1024),)
        _randn_fill_triton[grid_fill](grad_output, total_hs, BLOCK=1024)
        _randn_fill_triton[grid_fill](hidden_states, total_hs, BLOCK=1024)

        # router_weight: [n_routed_experts, hidden_size], bfloat16
        router_weight = torch.empty((n_routed_experts, H), dtype=torch.bfloat16, device=device)
        _randn_fill_triton[(n_routed_experts * H,)](router_weight, n_routed_experts * H, BLOCK=1024)

        # shared_expert weights: [m, hidden_size] for m in {gate, up, down},


def run(*args):
    return ModelNew()(*args)
