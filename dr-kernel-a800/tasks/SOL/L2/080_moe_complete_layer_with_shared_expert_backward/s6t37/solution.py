import torch
import triton
import triton.language as tl


@triton.jit
def linear_rowwise_bf16_to_f32(
    X_ptr,         # *bfloat16, shape [M, K]
    W_ptr,         # *bfloat16, shape [K, N]
    Y_ptr,         # *float32,  shape [M, N]
    M: tl.constexpr,  # int (compile-time for Triton)
    K: tl.constexpr,  # int
    N: tl.constexpr,  # int
    stride_xm,     # int
    stride_xk,     # int
    stride_wk,     # int (row stride of W: W[k, n] increments by stride_wk per k)
    stride_wn,     # int (col stride of W: per n step)
    stride_ym,     # int
    stride_yn,     # int
    BLOCK_K: tl.constexpr,  # tile size for K
):
    # Each program handles one row m
    m = tl.program_id(0)
    # If m >= M, exit (usually grid ensures m<M)
    if m >= M:
        return

    # Accumulator for this row
    acc = tl.zeros((N,), dtype=tl.float32)

    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)  # vector [BLOCK_K]
        # Mask for valid k in range
        mask_k = k_idx < K

        # Load X[m, k] vector of length BLOCK_K
        x_vec = tl.load(X_ptr + m * stride_xm + k_idx * stride_xk, mask=mask_k, other=0.0)
        x_vec = x_vec.to(tl.float32)  # promote to fp32 for accumulation

        # Load W[k, :] tile across N (vector of length N)
        # We iterate over k_idx to form a 2D load: [BLOCK_K, N]
        acc_tile = tl.zeros((BLOCK_K, N), dtype=tl.float32)
        for di in range(BLOCK_K):
            k = k0 + di
            if k < K:
                w_row_ptr = W_ptr + k * stride_wk
                w_vals = tl.load(w_row_ptr + tl.arange(0, N) * stride_wn, mask=tl.arange(0, N) < N, other=0.0)
                acc_tile[di, :] = w_vals.to(tl.float32)

        # Accumulate: acc += sum over di of x_vec[di] * acc_tile[di, :]
        # Implement as dot product of x_vec with each row (broadcasted multiply and sum)
        # Note: acc is vector [N]; acc_tile[:, n] can be broadcast with x_vec
        # Unrolled per di
        for di in range(BLOCK_K):
            k = k0 + di
            if k < K:
                w_row_ptr = W_ptr + k * stride_wk
                w_vals = tl.load(w_row_ptr + tl.arange(0, N) * stride_wn, mask=tl.arange(0, N) < N, other=0.0).to(tl.float32)
                acc += x_vec[di] * w_vals

    # Store acc to Y[m, :]
    store_idx = tl.arange(0, N)
    store_mask = store_idx < N
    tl.store(Y_ptr + m * stride_ym + store_idx * stride_yn, acc, mask=store_mask)


@triton.jit
def silu_mul_kernel(
    A_ptr, # *float32, shape [M, N]
    B_ptr, # *float32, shape [M, N]
    Y_ptr, # *float32, shape [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_ym, stride_yn,
):
    pid = tl.program_id(0)
    total = M * N
    if pid >= total:
        return
    # Map pid to (m, n)
    m = pid // N
    n = pid % N
    # Compute values
    a = tl.load(A_ptr + m * stride_am + n * stride_an)
    b = tl.load(B_ptr + m * stride_bm + n * stride_bn)
    # y = a * sigmoid(a) * b
    y = a * tl.sigmoid(a) * b
    tl.store(Y_ptr + m * stride_ym + n * stride_yn, y)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, shared_expert_gate_weight, shared_expert_up_weight):
        """
        Triton-only forward that computes:
        gate_out = hidden_states @ shared_expert_gate_weight.T  (bfloat16)
        up_out   = hidden_states @ shared_expert_up_weight.T    (bfloat16)
        shared_activated = SiLU(gate_out) * up_out             (bfloat16 returned)
        """
        # Ensure CUDA and contiguous
        device = hidden_states.device
        assert device.type == "cuda", "ModelNew expects CUDA tensors"

        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        # Both weights are [K, N]
        assert shared_expert_gate_weight.shape == (K, 1408), "gate_weight must be [K, 1408]"
        assert shared_expert_up_weight.shape == (K, 1408), "up_weight must be [K, 1408]"
        N = 1408

        # Prepare outputs as float32 (accumulation)
        gate_out = torch.empty((M, N), dtype=torch.float32, device=device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=device)
        activated = torch.empty((M, N), dtype=torch.float32, device=device)

        # Make inputs contiguous for simple stride math
        X = hidden_states.contiguous()                       # [M, K], bfloat16
        W_gate = shared_expert_gate_weight.contiguous()     # [K, N], bfloat16
        W_up = shared_expert_up_weight.contiguous()         # [K, N], bfloat16

        # Launch GEMV for gate_out: Y = X @ W_gate^T
        grid1 = (M,)
        linear_rowwise_bf16_to_f32[grid1](
            X, W_gate, gate_out,
            M, K, N,
            X.stride(0), X.stride(1),
            W_gate.stride(0), W_gate.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_K=64,  # tile size for K
            num_warps=4, num_stages=2
        )

        # Launch GEMV for up_out: Y = X @ W_up^T
        up_out = torch.empty((M, N), dtype=torch.float32, device=device)  # re-use output tensor
        linear_rowwise_bf16_to_f32[grid1](
            X, W_up, up_out,
            M, K, N,
            X.stride(0), X.stride(1),
            W_up.stride(0), W_up.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        # Elementwise compute: y = gate_out * sigmoid(gate_out) * up_out
        grid2 = (M * N,)
        silu_mul_kernel[grid2](
            gate_out, up_out, activated,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated.stride(0), activated.stride(1),
            num_warps=4, num_stages=1
        )

        # Return as bfloat16 (final cast not a torch op on tensor; it's a dtype conversion)
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
