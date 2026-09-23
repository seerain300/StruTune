import torch
import triton
import triton.language as tl


@triton.jit
def linear_rowwise_bf16_to_f32(
    X_ptr,  # *bfloat16, shape [M, K]
    W_ptr,  # *bfloat16, shape [K, N]
    Y_ptr,  # *float32,  shape [M, N]
    M: tl.constexpr,  # int
    K: tl.constexpr,  # int
    N: tl.constexpr,  # int
    stride_xm,        # int
    stride_xk,        # int
    stride_wk,        # int
    stride_wn,        # int
    stride_ym,        # int
    stride_yn,        # int
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(axis=0)  # one program per row
    # Accumulator for output row m
    acc = tl.zeros((N,), dtype=tl.float32)
    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        # Load X[m, k:k+BLOCK_K] (bfloat16) and promote to float32
        x = tl.load(
            X_ptr + m * stride_xm + k_offsets * stride_xk,
            mask=k_offsets < K,
            other=0.0,
        ).to(tl.float32)  # [BLOCK_K], float32

        # Load W[k:k+BLOCK_K, :] (bfloat16) and promote to float32
        n_offsets = tl.arange(0, N)
        w = tl.load(
            W_ptr + k_offsets[:, None] * stride_wk + n_offsets[None, :] * stride_wn,
            mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N),
            other=0.0,
        ).to(tl.float32)  # [BLOCK_K, N], float32

        # Accumulate: acc += sum_k x[k] * w[k, :]
        acc += tl.sum(w * x[:, None], axis=0)

    # Store acc to Y[m, :]
    tl.store(Y_ptr + m * stride_ym + n_offsets * stride_yn, acc, mask=n_offsets < N)


@triton.jit
def silu_mul_kernel(
    GateOut_ptr,    # *float32, shape [M, N]
    Up_ptr,         # *float32, shape [M, N]
    Out_ptr,        # *float32, shape [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_om, stride_on,
):
    pid = tl.program_id(axis=0)  # linearize over M*N
    m = pid // N
    n = pid % N
    if (m < M) & (n < N):
        go = tl.load(GateOut_ptr + m * stride_gm + n * stride_gn)  # float32
        up = tl.load(Up_ptr + m * stride_um + n * stride_un)       # float32
        # y = go * sigmoid(go) * up
        y = go * tl.sigmoid(go) * up
        tl.store(Out_ptr + m * stride_om + n * stride_on, y)


def _triton_linear_rowwise_bf16_to_f32(X, W):
    """
    Compute Y = X @ W^T with X[M, K] bfloat16 and W[K, N] bfloat16, returns Y[M, N] float32.
    """
    M, K = X.shape
    K_w, N = W.shape
    assert K == K_w, "Incompatible shapes for X and W in linear_rowwise_bf16_to_f32."
    Xc = X.contiguous()
    Wc = W.contiguous()
    Y = torch.empty((M, N), dtype=torch.float32, device=X.device)
    BLOCK_K = 256
    grid = (M,)
    linear_rowwise_bf16_to_f32[grid](
        Xc, Wc, Y,
        M, K, N,
        Xc.stride(0), Xc.stride(1),
        Wc.stride(0), Wc.stride(1),
        Y.stride(0), Y.stride(1),
        BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return Y


def _triton_silu_mul(gate_out, up):
    """
    Elementwise: out = gate_out * sigmoid(gate_out) * up
    gate_out, up: [M, N] float32 tensors
    """
    M, N = gate_out.shape
    out = torch.empty((M, N), dtype=torch.float32, device=gate_out.device)
    grid = (M * N,)
    silu_mul_kernel[grid](
        gate_out, up, out,
        M, N,
        gate_out.stride(0), gate_out.stride(1),
        up.stride(0), up.stride(1),
        out.stride(0), out.stride(1),
        num_warps=4,
        num_stages=1,
    )
    return out


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, shared_expert_gate_weight, shared_expert_up_weight):
        """
        Triton-only forward:
        - Computes gate = hidden_states @ gate_weight.T
        - Computes up = hidden_states @ up_weight.T
        - Returns shared_activated = SiLU(gate) * up, as bfloat16.
        """
        # Ensure tensors are CUDA
        assert hidden_states.is_cuda, "ModelNew requires CUDA tensors."
        assert shared_expert_gate_weight.is_cuda and shared_expert_up_weight.is_cuda, "All tensors must be on CUDA."

        # Use bfloat16 inputs for GEMV kernels
        hidden_states = hidden_states.to(torch.bfloat16).contiguous()
        gate_weight = shared_expert_gate_weight.to(torch.bfloat16).contiguous()
        up_weight = shared_expert_up_weight.to(torch.bfloat16).contiguous()

        M, K = hidden_states.shape
        _, N = gate_weight.shape
        assert gate_weight.shape[0] == K and up_weight.shape[0] == K, "Weight first dim must match hidden_size."
        assert up_weight.shape[1] == N, "Weights must have matching N."

        # Compute gate_out and up_out as float32
        gate_out = _triton_linear_rowwise_bf16_to_f32(hidden_states, gate_weight)  # [M, N], float32
        up_out = _triton_linear_rowwise_bf16_to_f32(hidden_states, up_weight)     # [M, N], float32

        # Compute SiLU(gate_out) * up_out
        activated = _triton_silu_mul(gate_out, up_out)  # [M, N], float32

        # Return as bfloat16
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
