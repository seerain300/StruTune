import torch
import triton
import triton.language as tl


@triton.jit
def linear_rowwise_bf16_to_f32(
    X_ptr,  # *bfloat16, shape [M, K]
    W_ptr,  # *bfloat16, shape [K, N]
    Y_ptr,  # *float32,  shape [M, N]
    M,      # int (runtime)
    K,      # int (runtime)
    N,      # int (runtime)
    stride_xm,  # stride of X along rows
    stride_xk,  # stride of X along cols
    stride_wk,  # stride of W along rows (K dim)
    stride_wn,  # stride of W along cols (N dim)
):
    # Each program computes one row m
    m = tl.program_id(axis=0)
    if m >= M:
        return

    # Accumulator for this row across N columns (float32)
    acc = tl.zeros((N,), dtype=tl.float32)

    # Loop over K: compute X[m, k] * W[k, :]
    for k in range(0, K):
        # Load x_scalar: X[m, k] as bfloat16
        x_scalar = tl.load(X_ptr + m * stride_xm + k * stride_xk)
        x_scalar_f32 = x_scalar.to(tl.float32)

        # Load W[k, :] as bfloat16 vector with mask
        offs_n = tl.arange(0, N)
        w_vec = tl.load(W_ptr + k * stride_wk + offs_n * stride_wn, mask=offs_n < N, other=0.0)

        # Accumulate: acc += x_scalar_f32 * w_vec_f32
        acc += x_scalar_f32 * w_vec.to(tl.float32)

    # Store acc to Y[m, :]
    tl.store(Y_ptr + m * N + offs_n, acc, mask=offs_n < N)


@triton.jit
def silu_mul_kernel(
    GateOut_ptr,  # *float32, shape [M, N]
    Up_ptr,       # *float32, shape [M, N]
    Y_ptr,        # *float32, shape [M, N]
    M,            # int
    N,            # int
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_ym, stride_yn,
):
    m = tl.program_id(axis=0)
    n = tl.program_id(axis=1)
    if m >= M or n >= N:
        return

    gate = tl.load(GateOut_ptr + m * stride_gm + n * stride_gn)
    up = tl.load(Up_ptr + m * stride_um + n * stride_un)

    # y = gate * sigmoid(gate) * up
    sig = 1.0 / (1.0 + tl.exp(-gate))
    y = gate * sig * up

    tl.store(Y_ptr + m * stride_ym + n * stride_yn, y)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Only use hidden_states and the two weights from get_inputs
        # Original signature is large, but we only need the following:
        hidden_states = args[0].contiguous()         # [M, K], bfloat16
        gate_weight = args[8].contiguous()           # [K, N], bfloat16
        up_weight = args[9].contiguous()             # [K, N], bfloat16

        M, K = hidden_states.shape
        K_w, N = gate_weight.shape
        assert K == K_w, f"hidden_states K={K} must match gate_weight rows K={K_w}"
        assert hidden_states.dtype == torch.bfloat16
        assert gate_weight.dtype == torch.bfloat16
        assert up_weight.dtype == torch.bfloat16

        # Output buffers (float32 for accumulation)
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)

        # Launch GEMV kernels: one program per row
        grid = (M,)
        linear_rowwise_bf16_to_f32[grid](
            hidden_states, gate_weight, gate_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            num_warps=2, num_stages=1
        )
        linear_rowwise_bf16_to_f32[grid](
            hidden_states, up_weight, up_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            num_warps=2, num_stages=1
        )

        # Elementwise SiLU * up
        activated = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        grid2 = (M, N)
        silu_mul_kernel[grid2](
            gate_out, up_out, activated,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated.stride(0), activated.stride(1),
            num_warps=2, num_stages=1
        )

        # Return as bfloat16 (cast via dtype constructor)
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
