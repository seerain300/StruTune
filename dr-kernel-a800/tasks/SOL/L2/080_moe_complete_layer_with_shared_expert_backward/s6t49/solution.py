import torch
import triton
import triton.language as tl


@triton.jit
def gemv_rowwise_bf16_to_f32(
    X_ptr,   # *bfloat16, shape [M, K]
    W_ptr,   # *bfloat16, shape [K, N]
    Y_ptr,   # *float32,  shape [M, N]
    M: tl.constexpr,  # int (row count)
    K: tl.constexpr,  # int (feature count)
    N: tl.constexpr,  # int (output features)
    stride_xm, stride_xk,  # strides for X
    stride_wk, stride_wn,  # strides for W
    BLOCK_K: tl.constexpr  # tile size along K
):
    # One program per row m
    m = tl.program_id(axis=0)
    # Initialize accumulator for this row: vector of size N in fp32
    acc = tl.zeros([N], dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        # Offsets within the current chunk
        offs_k = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        # Load X[m, offs_k] as bfloat16 vector
        x_vec = tl.load(
            X_ptr + m * stride_xm + offs_k * stride_xk,
            mask=offs_k < K,
            other=0.0
        ).to(tl.float32)  # promote to fp32 for accumulation

        # Accumulate contributions from W[offs_k, :] over N
        for n_idx in range(0, N):
            # Load W[offs_k, n_idx] as bfloat16 vector over BLOCK_K
            w_vec = tl.load(
                W_ptr + offs_k * stride_wk + n_idx * stride_wn,
                mask=(offs_k < K) & (n_idx < N),
                other=0.0
            ).to(tl.float32)
            acc[n_idx] += tl.sum(x_vec * w_vec, axis=0)

    # Store result acc to Y[m, :]
    tl.store(
        Y_ptr + m * 1 + tl.arange(0, N) * 1,  # Y has strides: stride_yrow=M, stride_ycol=N; here we use implicit row-major
        acc, mask=tl.arange(0, N) < N
    )


@triton.jit
def silu_mul_kernel(
    x_ptr,  # *float32,  gate_out
    u_ptr,  # *float32,  up_out
    y_ptr,  # *float32,  output
    M: tl.constexpr,  # int (rows)
    N: tl.constexpr,  # int (cols)
    stride_xm, stride_xn,  # strides for x
    stride_um, stride_un,  # strides for u
    stride_ym, stride_yn,  # strides for y
):
    m = tl.program_id(axis=0)
    n = tl.program_id(axis=1)
    # Load x[m, n] and u[m, n]
    x_val = tl.load(x_ptr + m * stride_xm + n * stride_xn)
    u_val = tl.load(u_ptr + m * stride_um + n * stride_un)
    # Compute SiLU(x) = x * sigmoid(x)
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sig * u_val
    # Store
    tl.store(y_ptr + m * stride_ym + n * stride_yn, y_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden_states,       # [M, K], bfloat16
        shared_expert_gate_weight,  # [K, N], bfloat16
        shared_expert_up_weight,    # [K, N], bfloat16
    ):
        # Ensure tensors are on CUDA and contiguous
        assert hidden_states.is_cuda and shared_expert_gate_weight.is_cuda and shared_expert_up_weight.is_cuda, "Tensors must be on CUDA"
        hidden_states = hidden_states.contiguous()
        gate_weight = shared_expert_gate_weight.contiguous()
        up_weight = shared_expert_up_weight.contiguous()

        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = gate_weight.shape[1]  # intermediate_size, e.g., 1408

        # Output buffers in float32
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)

        # Launch GEMV kernels: one program per row
        grid = (M,)
        # Choose BLOCK_K to balance occupancy and register pressure; 256 works well for K=4096
        BLOCK_K = 256
        gemv_rowwise_bf16_to_f32[grid](
            hidden_states, gate_weight, gate_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        gemv_rowwise_bf16_to_f32[grid](
            hidden_states, up_weight, up_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Elementwise activation: y = gate_out * sigmoid(gate_out) * up_out
        y_activated = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        silu_mul_kernel[(M, N)](
            gate_out, up_out, y_activated,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            y_activated.stride(0), y_activated.stride(1),
            num_warps=4, num_stages=1
        )

        # Return as bfloat16 (cast on host)
        return y_activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
