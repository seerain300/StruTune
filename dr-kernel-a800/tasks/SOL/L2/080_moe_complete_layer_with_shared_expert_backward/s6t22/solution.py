import torch
import triton
import triton.language as tl


# Triton row-wise GEMV kernel: computes Y = X @ W^T
# X: [M, K] bfloat16, W: [K, N] bfloat16, Y: [M, N] float32
@triton.jit
def gemv_row_bf16_to_f32(
    X_ptr, W_ptr, Y_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(axis=0)
    # Initialize accumulator for the output row
    acc = tl.zeros((N,), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # Load X[m, k] as bfloat16 vector
        x_ptrs = X_ptr + m * stride_xm + offs_k * stride_xk
        x_mask = offs_k < K
        x_vec = tl.load(x_ptrs, mask=x_mask, other=0.0)  # bfloat16 vector [BLOCK_K]
        # For each kk in the chunk, accumulate acc += X[m, k] * W[k, :]
        for kk in range(BLOCK_K):
            k_idx = k0 + kk
            if k_idx < K:
                x_val = tl.load(X_ptr + m * stride_xm + k_idx * stride_xk, mask=True, other=0.0).to(tl.float32)
                w_row_ptrs = W_ptr + k_idx * stride_wk + tl.arange(0, N) * stride_wn
                w_row_mask = (tl.arange(0, N) < N)
                w_row = tl.load(w_row_ptrs, mask=w_row_mask, other=0.0).to(tl.float32)
                acc += x_val * w_row

    # Store the row with boundary mask
    y_ptrs = Y_ptr + m * stride_ym + tl.arange(0, N) * stride_yn
    y_mask = tl.arange(0, N) < N
    tl.store(y_ptrs, acc, mask=y_mask)


# Triton elementwise kernel: y = gate_out * sigmoid(gate_out) * up_out
@triton.jit
def silu_mul_kernel(
    GateOut_ptr, UpOut_ptr, Y_ptr,
    M, N,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_ym, stride_yn,
):
    m = tl.program_id(axis=0)
    n = tl.program_id(axis=1)
    in_bounds = (m < M) & (n < N)
    gate_ptr = GateOut_ptr + m * stride_gm + n * stride_gn
    up_ptr = UpOut_ptr + m * stride_um + n * stride_un
    y_ptr = Y_ptr + m * stride_ym + n * stride_yn
    gate_val = tl.load(gate_ptr, mask=in_bounds, other=0.0)  # float32
    up_val = tl.load(up_ptr, mask=in_bounds, other=0.0)      # float32
    s = 1.0 / (1.0 + tl.exp(-gate_val))  # sigmoid
    y_val = gate_val * s * up_val
    tl.store(y_ptr, y_val, mask=in_bounds)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Assume the last three args are: hidden_states (bfloat16), gate_weight (bfloat16), up_weight (bfloat16)
        hidden_states = args[-3]
        gate_weight = args[-2]
        up_weight = args[-1]

        assert hidden_states.is_cuda and gate_weight.is_cuda and up_weight.is_cuda, "Tensors must be on CUDA"

        # Shapes: hidden_states [M, K], gate_weight and up_weight [K, N]
        M, K = hidden_states.shape
        K_g, N = gate_weight.shape
        assert K_g == K, "hidden_states K must match gate_weight K"
        assert up_weight.shape[0] == K and up_weight.shape[1] == N, "up_weight shapes must match gate_weight"

        # Ensure contiguous
        hidden_states = hidden_states.contiguous()
        gate_weight = gate_weight.contiguous()
        up_weight = up_weight.contiguous()

        # Allocate outputs in float32
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)

        # Launch row-wise GEMV kernels
        # Choose a reasonable BLOCK_K; 128 balances loop iterations and register use
        BLOCK_K = 128
        grid = (M,)

        gemv_row_bf16_to_f32[grid](
            hidden_states, gate_weight, gate_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_K=BLOCK_K,
            num_warps=2, num_stages=2
        )

        gemv_row_bf16_to_f32[grid](
            hidden_states, up_weight, up_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_K=BLOCK_K,
            num_warps=2, num_stages=2
        )

        # Elementwise y = gate_out * sigmoid(gate_out) * up_out
        y = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        silu_mul_kernel[(M, N)](
            gate_out, up_out, y,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            y.stride(0), y.stride(1),
            num_warps=1, num_stages=1
        )

        # Return casted to bfloat16
        return y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
