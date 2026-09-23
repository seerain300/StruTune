import torch
import triton
import triton.language as tl


@triton.jit
def bf16_rowwise_bf16_to_f32(
    X_ptr,   # *bfloat16, shape [M, K]
    W_ptr,   # *bfloat16, shape [K, N]
    Y_ptr,   # *float32,  shape [M, N]
    M, K, N,  # int32
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_K: tl.constexpr,
):
    # One program per row m
    m = tl.program_id(0)
    # Accumulator for this row: length N in float32
    acc = tl.zeros((N,), dtype=tl.float32)
    # Loop over K dimension in tiles
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        # Load x_vec: X[m, k_offsets]
        x_ptrs = X_ptr + m * stride_xm + k_offsets * stride_xk
        # Mask for valid k
        x_mask = (k_offsets < K) & (m < M)
        x_vec = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)  # [BLOCK_K], float32

        # Load W_tile: W[k_offsets, :]
        w_ptrs = W_ptr + k_offsets[:, None] * stride_wk + tl.arange(0, N)[None, :] * stride_wn
        k_mask = k_offsets < K
        w_mask = (k_mask[:, None]) & (tl.arange(0, N)[None, :] < N)
        W_tile = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)  # [BLOCK_K, N], float32

        # Accumulate: acc += sum_k x_vec[k] * W_tile[k, :]
        acc += tl.sum(W_tile * x_vec[:, None], axis=0)  # reduce over K tile -> [N]

    # Store acc to Y[m, :]
    y_ptrs = Y_ptr + m * stride_ym + tl.arange(0, N) * stride_yn
    y_mask = (m < M) & (tl.arange(0, N) < N)
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def silu_mul_elementwise(
    Gate_ptr,  # *float32, shape [M, N]
    Up_ptr,    # *float32, shape [M, N]
    Out_ptr,   # *float32, shape [M, N]
    M, N,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_om, stride_on,
):
    # 1D grid over M*N
    pid = tl.program_id(0)
    m = pid // N
    n = pid % N
    mask = (m < M) & (n < N)

    gm = Gate_ptr + m * stride_gm + n * stride_gn
    um = Up_ptr + m * stride_um + n * stride_un
    om = Out_ptr + m * stride_om + n * stride_on

    gate = tl.load(gm, mask=mask, other=0.0)
    up = tl.load(um, mask=mask, other=0.0)
    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-gate))
    out = gate * sig * up
    tl.store(om, out, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states,          # [M, K], bfloat16
        shared_expert_gate_weight,  # [K, N], bfloat16
        shared_expert_up_weight,     # [K, N], bfloat16
    ):
        # Ensure on CUDA and contiguous
        assert hidden_states.is_cuda, "Inputs must be on CUDA device for Triton kernels."
        hidden_states = hidden_states.contiguous()
        gate_weight = shared_expert_gate_weight.contiguous()
        up_weight = shared_expert_up_weight.contiguous()

        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = gate_weight.shape[1]  # N is 1408 in provided inputs

        # Output buffers (float32 for accumulation, cast at end)
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)

        # Launch row-wise GEMV kernels
        BLOCK_K = 128
        grid = (M,)
        bf16_rowwise_bf16_to_f32[grid](
            hidden_states, gate_weight, gate_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        bf16_rowwise_bf16_to_f32[grid](
            hidden_states, up_weight, up_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Elementwise SiLU product
        activated = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        silu_mul_elementwise[(M * N,)](
            gate_out, up_out, activated,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated.stride(0), activated.stride(1),
            num_warps=4, num_stages=2
        )

        # Return bfloat16 (dtype constructor, no torch op on tensors)
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
