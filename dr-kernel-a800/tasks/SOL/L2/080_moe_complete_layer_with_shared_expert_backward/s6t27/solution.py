import torch
import triton
import triton.language as tl


@triton.jit
def linear_rowwise_bf16_to_f32(
    X_ptr,            # *bfloat16, shape [M, K]
    W_ptr,            # *bfloat16, shape [K, N]
    Y_ptr,            # *float32,  shape [M, N]
    M: tl.constexpr,  # int
    K: tl.constexpr,  # int
    N: tl.constexpr,  # int
    stride_xm,        # int
    stride_xk,        # int
    stride_wk,        # int
    stride_wn,        # int
    stride_ym,        # int
    stride_yn,        # int
    BLOCK_K: tl.constexpr,  # tile size along K
    BLOCK_N: tl.constexpr,  # tile size along N (vector length)
):
    # One program per row m
    m = tl.program_id(0)

    # Accumulator vector for this row
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        # Load X[m, k0:k0+BLOCK_K] as a vector
        k_vec = k0 + tl.arange(0, BLOCK_K)
        x_vec = tl.load(
            X_ptr + m * stride_xm + k_vec * stride_xk,
            mask=k_vec < K,
            other=0.0,
        ).to(tl.float32)  # promote to float32

        # Load W[k0:k0+BLOCK_K, :] as a [BLOCK_K, BLOCK_N] tile
        n_vec = tl.arange(0, BLOCK_N)
        w_tile = tl.load(
            W_ptr + k_vec[:, None] * stride_wk + n_vec[None, :] * stride_wn,
            mask=(k_vec[:, None] < K) & (n_vec[None, :] < N),
            other=0.0,
        ).to(tl.float32)  # promote to float32

        # Accumulate: acc += sum_k x_vec[k] * w_tile[k, :]
        # Equivalent to acc += tl.sum(x_vec[:, None] * w_tile, axis=0)
        acc += tl.sum(x_vec[:, None] * w_tile, axis=0)

    # Store acc to Y[m, :]
    tl.store(
        Y_ptr + m * stride_ym + n_vec * stride_yn,
        acc,
        mask=n_vec < N,
    )


@triton.jit
def silu_mul_kernel(
    Gate_ptr,     # *float32, shape [M, N]
    Up_ptr,       # *float32, shape [M, N]
    Y_ptr,        # *float32, shape [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_ym, stride_yn,
):
    pid = tl.program_id(0)
    # 1D launch: grid=(M*N,)
    total = M * N
    m = pid // N
    n = pid % N

    # Bounds check
    if m < M and n < N:
        g = tl.load(Gate_ptr + m * stride_gm + n * stride_gn)
        u = tl.load(Up_ptr + m * stride_um + n * stride_un)
        # sigmoid(g) = 1 / (1 + exp(-g))
        sig = 1.0 / (1.0 + tl.exp(-g))
        y = g * sig * u
        tl.store(Y_ptr + m * stride_ym + n * stride_yn, y)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args are: grad_output, hidden_states, router_weight, e_score_correction_bias, 
        # router_logits, scores, topk_indices, topk_weights, score_mask,
        # shared_expert_gate_weight, shared_expert_up_weight, shared_expert_down_weight,
        # shared_gate_output, shared_up_output, shared_activated
        # We only need hidden_states, shared_expert_gate_weight, shared_expert_up_weight.

        hidden = args[1]             # [M, K] bfloat16
        gate_w = args[9]             # [K, N] bfloat16
        up_w = args[10]              # [K, N] bfloat16

        # Ensure CUDA tensors and contiguity
        device = hidden.device
        assert hidden.is_cuda and gate_w.is_cuda and up_w.is_cuda, "Tensors must be on CUDA device"
        hidden = hidden.contiguous()
        gate_w = gate_w.contiguous()
        up_w = up_w.contiguous()

        M = hidden.shape[0]
        K = hidden.shape[1]
        N = gate_w.shape[1]  # N = 1408 in provided inputs

        # Output buffers for GEMV
        gate_out = torch.empty((M, N), dtype=torch.float32, device=device)  # [M, N] float32
        up_out = torch.empty((M, N), dtype=torch.float32, device=device)    # [M, N] float32

        # Launch GEMV kernel: one program per row
        grid = (M,)
        linear_rowwise_bf16_to_f32[grid](
            hidden, gate_w, gate_out,
            M, K, N,
            hidden.stride(0), hidden.stride(1),
            gate_w.stride(0), gate_w.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_K=128, BLOCK_N=128,
            num_warps=4, num_stages=2
        )

        linear_rowwise_bf16_to_f32[grid](
            hidden, up_w, up_out,
            M, K, N,
            hidden.stride(0), hidden.stride(1),
            up_w.stride(0), up_w.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_K=128, BLOCK_N=128,
            num_warps=4, num_stages=2
        )

        # Elementwise compute: activated = gate_out * sigmoid(gate_out) * up_out
        activated = torch.empty((M, N), dtype=torch.float32, device=device)
        # 1D grid
        total = M * N
        silu_mul_kernel[(total,)](
            gate_out, up_out, activated,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated.stride(0), activated.stride(1),
            num_warps=1, num_stages=1
        )

        # Cast to bfloat16 and return
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
