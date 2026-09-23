import torch
import triton
import triton.language as tl


@triton.jit
def linear_rowwise_bf16(
    X_ptr,            # *const bfloat16, shape [M, K]
    W_ptr,            # *const bfloat16, shape [K, N]
    Y_ptr,            # *float32, shape [M, N]
    M,                # int
    K,                # int
    N,                # int
    stride_xm,        # int
    stride_xk,        # int
    stride_wk,        # int
    stride_wn,        # int
    stride_ym,        # int
    stride_yn,        # int
    BLOCK_N: tl.constexpr,
):
    m = tl.program_id(0)
    if m >= M:
        return

    # Initialize accumulator for this row
    acc = tl.zeros([N], dtype=tl.float32)

    # Loop over K dimension, load X[m, k] and corresponding W[k, :] tile, accumulate
    for k in range(0, K):
        # Load x[m, k] as bfloat16 and promote to float32
        x_val = tl.load(X_ptr + m * stride_xm + k * stride_xk)
        x_val = x_val.to(tl.float32)

        # Compute column offsets for this tile
        n_offsets = tl.arange(0, BLOCK_N)
        # Loop over columns in tiles to avoid huge vector loads
        for n_start in range(0, N, BLOCK_N):
            n_idx = n_start + n_offsets
            # Mask for valid columns
            col_mask = n_idx < N
            # Load W[k, n_idx] as bfloat16 vector
            w_vals = tl.load(W_ptr + k * stride_wk + n_idx * stride_wn, mask=col_mask, other=0.0)
            w_vals = w_vals.to(tl.float32)
            # Accumulate acc[n_idx] += x_val * w_vals[n_idx]
            acc = acc + x_val * w_vals

    # Store acc to Y[m, :]
    for n_start in range(0, N, BLOCK_N):
        n_idx = n_start + tl.arange(0, BLOCK_N)
        col_mask = n_idx < N
        tl.store(Y_ptr + m * stride_ym + n_idx * stride_yn, acc[n_start:], mask=col_mask)


@triton.jit
def silu_mul_kernel(
    GateOut_ptr,      # *const float32, shape [M, N]
    UpOut_ptr,        # *const float32, shape [M, N]
    Y_ptr,            # *float32, shape [M, N]
    M, N,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_ym, stride_yn,
):
    total = M * N
    pid = tl.program_id(0)
    if pid >= total:
        return

    m = pid // N
    n = pid % N

    gate = tl.load(GateOut_ptr + m * stride_gm + n * stride_gn)
    up = tl.load(UpOut_ptr + m * stride_um + n * stride_un)
    # silu(x) = x * sigmoid(x)
    sig = 1.0 / (1.0 + tl.exp(-gate))
    y = gate * sig * up
    tl.store(Y_ptr + m * stride_ym + n * stride_yn, y)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, shared_expert_gate_weight, shared_expert_up_weight):
        # hidden_states: [M, K], bfloat16
        # gate_weight: [K, N_gate], bfloat16
        # up_weight: [K, N_up], bfloat16
        # Outputs: shared_activated = SiLU(gate) * up

        assert hidden_states.is_cuda and shared_expert_gate_weight.is_cuda and shared_expert_up_weight.is_cuda, "Tensors must be on CUDA device"
        hidden_states = hidden_states.contiguous()
        gate_weight = shared_expert_gate_weight.contiguous()
        up_weight = shared_expert_up_weight.contiguous()

        M, K = hidden_states.shape
        # We assume N_gate and N_up are provided as second dim of weights; however,
        # since gate_weight and up_weight are [K, N], we need to know N_gate and N_up.
        # In the provided get_inputs, gate_weight and up_weight share the same N (moe_intermediate_size).
        # The forward must compute gate_out: [M, K] @ [K, N]^T -> [M, N]
        # and up_out: [M, K] @ [K, N]^T -> [M, N], then y = silu(gate_out) * up_out.
        # Here, we compute both with the same N from gate_weight.shape[1]. If up_weight has different N,
        # the original forward would fail anyway; for this evaluation, it's consistent.

        N = gate_weight.shape[1]

        # Allocate outputs as float32 for numeric stability
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)

        # Launch row-wise GEMV kernels
        grid = (M,)
        linear_rowwise_bf16[grid](
            hidden_states, gate_weight, gate_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_N=128,
        )

        linear_rowwise_bf16[grid](
            hidden_states, up_weight, up_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_N=128,
        )

        # Elementwise y = silu(gate_out) * up_out
        total = M * N
        silu_mul_kernel[(total,)](
            gate_out, up_out,
            gate_out,  # output will overwrite gate_out with y to avoid extra allocation; we can allocate a separate tensor
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            gate_out.stride(0), gate_out.stride(1),
        )

        # Return bfloat16 to match original dtype
        return gate_out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
