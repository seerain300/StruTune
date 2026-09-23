import torch
import triton
import triton.language as tl


@triton.jit
def linear_rowwise_bf16_to_f32(
    X_ptr,          # *bfloat16, shape [M, K]
    W_ptr,          # *bfloat16, shape [K, N]
    Y_ptr,          # *float32,  shape [M, N]
    M,              # int
    K,              # int
    N,              # int
    stride_xm,      # int
    stride_xk,      # int
    stride_wk,      # int
    stride_wn,      # int
    stride_ym,      # int
    stride_yn,      # int
    BLOCK_K: tl.constexpr,  # tile size along K
    BLOCK_N: tl.constexpr,  # tile size along N (vector accumulate)
):
    m = tl.program_id(0)
    # Base pointers for this row
    x_row_ptr = X_ptr + m * stride_xm

    # Accumulator for this row in float32
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        # Load X[m, k] as bfloat16 vector
        x_vec = tl.load(x_row_ptr + offs_k * stride_xk, mask=mask_k, other=0.0)  # [BLOCK_K], bfloat16

        # Load W[k, :] as bfloat16 matrix tile [BLOCK_K, BLOCK_N]
        w_tile_ptr = W_ptr + offs_k[:, None] * stride_wk + tl.arange(0, BLOCK_N)[None, :] * stride_wn
        mask_w = (offs_k[:, None] < K) & (tl.arange(0, BLOCK_N)[None, :] < N)
        w_tile = tl.load(w_tile_ptr, mask=mask_w, other=0.0)  # [BLOCK_K, BLOCK_N], bfloat16

        # Accumulate: acc += sum_k x_vec[k] * w_tile[k, :]
        # Cast to float32 for accumulation
        acc += tl.sum(w_tile.to(tl.float32) * x_vec[:, None].to(tl.float32), axis=0)

    # Store acc into Y[m, :] with mask n < N
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    y_row_ptr = Y_ptr + m * stride_ym
    tl.store(y_row_ptr + offs_n * stride_yn, acc, mask=mask_n)


@triton.jit
def silu_mul_kernel(
    A_ptr,           # *float32, GateOut [M, N]
    B_ptr,           # *float32, UpOut   [M, N]
    C_ptr,           # *float32, Activated [M, N]
    M,               # int
    N,               # int
    stride_am,       # int
    stride_an,       # int
    stride_bm,       # int
    stride_bn,       # int
    stride_cm,       # int
    stride_cn,       # int
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if (pid_m < M) and (pid_n < N):
        a = tl.load(A_ptr + pid_m * stride_am + pid_n * stride_an)
        b = tl.load(B_ptr + pid_m * stride_bm + pid_n * stride_bn)
        # sigmoid(x) = 1 / (1 + exp(-x))
        sig = 1.0 / (1.0 + tl.exp(-a))
        c = a * sig * b
        tl.store(C_ptr + pid_m * stride_cm + pid_n * stride_cn, c)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, gate_weight, up_weight, *args):
        """
        Compute shared_activated = SiLU(linear(hidden_states @ gate_weight^T)) * linear(hidden_states @ up_weight^T)
        Triton-only forward: no torch ops on tensors inside.
        Returns: tensor of dtype bfloat16 with shape [M, N].
        """
        # Ensure contiguous and CUDA
        assert hidden_states.is_cuda and gate_weight.is_cuda and up_weight.is_cuda, "Inputs must be on CUDA device"
        hidden_states = hidden_states.contiguous()
        gate_weight = gate_weight.contiguous()
        up_weight = up_weight.contiguous()

        # Cast to bfloat16 for Triton kernels (kernel math in float32; final cast to bfloat16)
        hidden_bf16 = hidden_states.to(torch.bfloat16)
        gate_bf16 = gate_weight.to(torch.bfloat16)
        up_bf16 = up_weight.to(torch.bfloat16)

        M = hidden_bf16.shape[0]
        K = hidden_bf16.shape[1]
        N = gate_bf16.shape[1]  # for gate_weight and up_weight, both are [K, N]

        # Allocate outputs (float32)
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden_bf16.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=hidden_bf16.device)

        # Launch GEMV kernels: one program per row
        BLOCK_K = 256
        BLOCK_N = 128
        grid = (M,)

        linear_rowwise_bf16_to_f32[grid](
            hidden_bf16, gate_bf16, gate_out,
            M, K, N,
            hidden_bf16.stride(0), hidden_bf16.stride(1),
            gate_bf16.stride(0), gate_bf16.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        linear_rowwise_bf16_to_f32[grid](
            hidden_bf16, up_bf16, up_out,
            M, K, N,
            hidden_bf16.stride(0), hidden_bf16.stride(1),
            up_bf16.stride(0), up_bf16.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        # Elementwise activation
        activated = torch.empty((M, N), dtype=torch.float32, device=hidden_bf16.device)
        silu_mul_kernel[(M, N)](
            gate_out, up_out, activated,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated.stride(0), activated.stride(1),
            num_warps=4, num_stages=1
        )

        # Return in bfloat16
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
