import torch
import triton
import triton.language as tl


@triton.jit
def linear_rowwise_bf16_to_f32(
    X_ptr,            # *bfloat16, shape [M, K]
    W_ptr,            # *bfloat16, shape [K, N]
    Y_ptr,            # *float32,  shape [M, N]
    M,                # int
    K: tl.constexpr,  # int (compile-time for loop)
    N: tl.constexpr,  # int (compile-time for vector length)
    stride_xm,        # int
    stride_xk,        # int
    stride_wk,        # int (row stride of W, i.e., stride along K)
    stride_wn,        # int (col stride of W, i.e., stride along N)
    stride_ym,        # int
    stride_yn,        # int
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    m = tl.program_id(axis=0)  # one program per row
    # Initialize accumulator for this row: float32 vector of length N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        # Load X row slice as bfloat16 vector
        x_vec = tl.load(
            X_ptr + m * stride_xm + k_offsets * stride_xk,
            mask=k_offsets < K,
            other=0.0
        ).to(tl.float32)  # promote to float32 for accumulation

        # Load W tile [BLOCK_K, BLOCK_N] as bfloat16
        n_offsets = tl.arange(0, BLOCK_N)
        w_ptrs = W_ptr + k_offsets[:, None] * stride_wk + n_offsets[None, :] * stride_wn
        w_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        # Accumulate: acc += x_scalar * W_tile across the K tile dimension
        # x_vec shape: [BLOCK_K], w_tile: [BLOCK_K, BLOCK_N]
        # For each kk in tile:
        for kk in range(BLOCK_K):
            # scalar x from this tile
            xk = x_vec[kk]  # float32
            wk = w_tile[kk, :]  # [BLOCK_N], float32
            acc += xk * wk

    # Store result for this row, masking n < N
    y_ptrs = Y_ptr + m * stride_ym + n_offsets * stride_yn
    tl.store(y_ptrs, acc, mask=n_offsets < N)


@triton.jit
def silu_mul_kernel(
    GateOut_ptr,      # *float32, shape [M, N]
    UpOut_ptr,        # *float32, shape [M, N]
    Y_ptr,            # *float32, shape [M, N]
    M, N,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_ym, stride_yn,
    # 1D grid over M*N
):
    pid = tl.program_id(axis=0)
    # Map pid to (m, n)
    n_elems = N
    m = pid // n_elems
    n = pid % n_elems

    # Bounds check
    if m >= M:
        return

    g = tl.load(GateOut_ptr + m * stride_gm + n * stride_gn)
    u = tl.load(UpOut_ptr + m * stride_um + n * stride_un)
    s = 1.0 / (1.0 + tl.exp(-g))  # sigmoid(g)
    y = g * s * u
    tl.store(Y_ptr + m * stride_ym + n * stride_yn, y)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args come from get_inputs: [...], hidden_states, shared_expert_gate_weight, shared_expert_up_weight, ...
        # We only need hidden_states, gate_weight, up_weight
        hidden_states = args[1]  # shape [M, K]
        gate_weight = args[2]    # shape [K, N]
        up_weight = args[3]      # shape [K, N]

        # Ensure contiguous and dtype
        hidden_states = hidden_states.contiguous()
        gate_weight = gate_weight.contiguous()
        up_weight = up_weight.contiguous()

        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = gate_weight.shape[1]  # N=1408

        # Allocate outputs (float32 for compute)
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        y = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)

        # Launch GEMV kernels: one program per row
        grid = (M,)
        # Choose tile sizes; K=4096, BLOCK_K=256 => 16 iterations; N=1408, BLOCK_N=128
        BLOCK_K = 256
        BLOCK_N = 128
        linear_rowwise_bf16_to_f32[grid](
            hidden_states, gate_weight, gate_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )
        linear_rowwise_bf16_to_f32[grid](
            hidden_states, up_weight, up_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        # Elementwise SiLU and multiply
        # 1D grid over M*N
        total = M * N
        silu_mul_kernel[(total,)](
            gate_out, up_out, y,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            y.stride(0), y.stride(1),
            num_warps=4, num_stages=2
        )

        # Cast to bfloat16 for return
        return y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
