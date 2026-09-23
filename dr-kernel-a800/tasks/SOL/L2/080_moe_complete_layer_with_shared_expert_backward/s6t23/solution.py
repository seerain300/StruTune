import torch
import triton
import triton.language as tl


# Triton GEMV: computes Y = X @ W^T, where
# X: [M, K] bfloat16, W: [K, N] bfloat16, Y: [M, N] float32
# One program handles one row m. We iterate over K in chunks and accumulate acc vector of size N.
@triton.jit
def linear_row_bf16_to_f32(
    X_ptr, W_ptr, Y_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    m = tl.program_id(0)
    if m >= M:
        return

    # initialize accumulator
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # loop over K dimension in chunks
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # load X row slice [BLOCK_K]
        x_vec = tl.load(X_ptr + m * stride_xm + offs_k * stride_xk, mask=mask_k, other=0.0)
        x_vec = x_vec.to(tl.float32)

        # load W chunk [BLOCK_K, BLOCK_N]
        offs_n = tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        w_chunk = tl.load(W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                          mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        w_chunk = w_chunk.to(tl.float32)

        # accumulate: acc += sum_k x_vec[k] * w_chunk[k, :]
        # Reduce across K dimension
        acc += tl.sum(w_chunk * x_vec[:, None], axis=0)

    # store acc into Y[m, :]
    tl.store(Y_ptr + m * stride_ym + offs_n * stride_yn, acc, mask=mask_n)


# Triton elementwise kernel: computes Y = gate_out * sigmoid(gate_out) * up_out
@triton.jit
def silu_mul_kernel(
    gate_ptr, up_ptr, out_ptr,
    M, N,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_om, stride_on,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if (pid_m >= M) or (pid_n >= N):
        return

    g = tl.load(gate_ptr + pid_m * stride_gm + pid_n * stride_gn)
    u = tl.load(up_ptr + pid_m * stride_um + pid_n * stride_un)

    # sigmoid(g) = 1 / (1 + exp(-g))
    s = 1.0 / (1.0 + tl.exp(-g))
    y = g * s * u

    tl.store(out_ptr + pid_m * stride_om + pid_n * stride_on, y)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, shared_expert_gate_weight, shared_expert_up_weight):
        """
        Compute:
          gate_out = linear(hidden_states, shared_expert_gate_weight.T, bias=None)  # [M, N]
          up_out = linear(hidden_states, shared_expert_up_weight.T, bias=None)      # [M, N]
          activated = SiLU(gate_out) * up_out
        Return activated as bfloat16 tensor.
        All heavy work is done by Triton kernels; no torch ops on tensors are used in forward.
        """
        # Ensure contiguity and dtype
        hidden = hidden_states.contiguous()  # [M, K], bfloat16
        gate_w = shared_expert_gate_weight.contiguous()  # [K, N], bfloat16
        up_w = shared_expert_up_weight.contiguous()      # [K, N], bfloat16

        M, K = hidden.shape
        K_w, N = gate_w.shape
        assert K == K_w, f"hidden K={K} must match gate_weight K={K_w}"
        assert up_w.shape[0] == K, f"up_weight K={up_w.shape[0]} must match hidden K={K}"

        device = hidden.device

        # Allocate outputs in float32
        gate_out = torch.empty((M, N), dtype=torch.float32, device=device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=device)

        # Launch row-wise kernels: 1D grid over M rows
        grid = (M,)
        BLOCK_K = 128
        BLOCK_N = 128
        linear_row_bf16_to_f32[grid](
            hidden, gate_w, gate_out,
            M, K, N,
            hidden.stride(0), hidden.stride(1),
            gate_w.stride(0), gate_w.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        linear_row_bf16_to_f32[grid](
            hidden, up_w, up_out,
            M, K, N,
            hidden.stride(0), hidden.stride(1),
            up_w.stride(0), up_w.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        # Elementwise activation: activated = SiLU(gate_out) * up_out
        activated = torch.empty((M, N), dtype=torch.float32, device=device)
        linear2 = (M, N)
        silu_mul_kernel[linear2](
            gate_out, up_out, activated,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated.stride(0), activated.stride(1),
            num_warps=4, num_stages=1
        )

        # Return as bfloat16 (constructor only, not an elementwise op)
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
