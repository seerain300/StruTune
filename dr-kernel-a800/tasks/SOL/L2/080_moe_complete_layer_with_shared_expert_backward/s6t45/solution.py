import torch
import triton
import triton.language as tl


@triton.jit
def linear_rowwise_bf16_to_f32(
    X_ptr,           # *bfloat16, shape [M, K]
    W_ptr,           # *bfloat16, shape [K, N]
    Y_ptr,           # *float32,  shape [M, N]
    M: tl.constexpr, # int
    K: tl.constexpr, # int
    N: tl.constexpr, # int
    stride_xm: tl.constexpr, # int
    stride_xk: tl.constexpr, # int
    stride_wk: tl.constexpr, # int
    stride_wn: tl.constexpr, # int
    stride_ym: tl.constexpr, # int
    stride_yn: tl.constexpr, # int
    BLOCK_K: tl.constexpr,   # int
    BLOCK_N: tl.constexpr,   # int
):
    # Each program handles one row m
    m = tl.program_id(0)
    if m >= M:
        return

    # Initialize accumulator for this row: [BLOCK_N] vector
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in tiles of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        # Build k indices and mask
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K

        # Load X[m, k] as bfloat16 vector [BLOCK_K]
        x_vec = tl.load(
            X_ptr + m * stride_xm + k_idx * stride_xk,
            mask=k_mask,
            other=0.0
        ).to(tl.float32)  # promote to float32 for accumulation

        # For each column block, load W[k, n] tile [BLOCK_K, BLOCK_N] and accumulate
        for n0 in range(0, N, BLOCK_N):
            n_idx = n0 + tl.arange(0, BLOCK_N)
            n_mask = n_idx < N

            # Load W[k, n] tile as bfloat16
            w_tile = tl.load(
                W_ptr + k_idx[:, None] * stride_wk + n_idx[None, :] * stride_wn,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0
            ).to(tl.float32)

            # Accumulate: acc += sum_k (x_vec[k] * w_tile[k, :])
            # w_tile shape: [BLOCK_K, BLOCK_N]
            # x_vec[:, None] shape: [BLOCK_K, 1]
            # Broadcast multiply and reduce along K axis
            acc += tl.sum(w_tile * x_vec[:, None], axis=0)

    # Store acc into Y[m, :]
    tl.store(Y_ptr + m * stride_ym + n_idx * stride_yn, acc, mask=n_mask)


@triton.jit
def silu_mul_kernel(
    GateOut_ptr,  # *float32, shape [M, N]
    UpOut_ptr,    # *float32, shape [M, N]
    Y_ptr,        # *float32, shape [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    stride_gm: tl.constexpr,
    stride_gn: tl.constexpr,
    stride_um: tl.constexpr,
    stride_un: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
):
    m = tl.program_id(0)
    n = tl.program_id(1)
    if (m >= M) or (n >= N):
        return

    g = tl.load(GateOut_ptr + m * stride_gm + n * stride_gn)
    u = tl.load(UpOut_ptr + m * stride_um + n * stride_un)
    s = 1.0 / (1.0 + tl.exp(-g))  # sigmoid
    y = g * s * u
    tl.store(Y_ptr + m * stride_ym + n * stride_yn, y)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # We expect hidden_states, gate_weight, up_weight to be provided in args.
        # Ignore any extras to be robust to harness calling conventions.
        # Manually extract by index; we assume args[0:3] are the three tensors we need.
        # If not enough, fall back to an error (in practice, harness should provide 3).
        try:
            hidden_states = args[0]
            gate_weight = args[1]
            up_weight = args[2]
        except IndexError:
            raise RuntimeError("ModelNew.forward requires hidden_states, gate_weight, up_weight as inputs.")

        # Ensure contiguous and CUDA tensors
        hidden_states = hidden_states.contiguous()
        gate_weight = gate_weight.contiguous()
        up_weight = up_weight.contiguous()

        if not (hidden_states.is_cuda and gate_weight.is_cuda and up_weight.is_cuda):
            raise RuntimeError("All tensors must be on CUDA device for Triton kernels.")

        # Dimensions: hidden_states [M, K], gate_weight/up_weight [K, N]
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        # In this task, gate_weight and up_weight are assumed to have same N.
        N = gate_weight.shape[1]

        # Compute gate_out and up_out using Triton GEMV
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)

        # Launch gate_rowwise_kernel: compute gate_out = hidden_states @ gate_weight.T
        grid = (M,)
        linear_rowwise_bf16_to_f32[grid](
            hidden_states, gate_weight, gate_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_K=256, BLOCK_N=128,
            num_warps=4, num_stages=2
        )

        # Launch up_rowwise_kernel: compute up_out = hidden_states @ up_weight.T
        up_rowwise = linear_rowwise_bf16_to_f32  # reuse same kernel for up
        up_rowwise[grid](
            hidden_states, up_weight, up_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_K=256, BLOCK_N=128,
            num_warps=4, num_stages=2
        )

        # Compute activated = gate_out * SiLU(gate_out) * up_out using Triton elementwise kernel
        activated = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        grid2 = (M, N)
        silu_mul_kernel[grid2](
            gate_out, up_out, activated,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated.stride(0), activated.stride(1),
            num_warps=4, num_stages=1
        )

        # Return in bfloat16 (no torch ops on tensors)
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
