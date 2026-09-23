import torch
import triton
import triton.language as tl


@triton.jit
def linear_rowwise_bf16_to_f32(
    X_ptr,   # *bfloat16, shape [M, K]
    W_ptr,   # *bfloat16, shape [K, N]
    Y_ptr,   # *float32,  shape [M, N]
    M,       # int32
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
    # One program per row m
    m = tl.program_id(0)

    # Accumulator for this row (float32 vector of length N)
    acc = tl.zeros((N,), dtype=tl.float32)

    # Loop over K dimension in tiles of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        # Load x_vec: X[m, k0:k0+BLOCK_K]
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        x_vec = tl.load(
            X_ptr + m * stride_xm + k_offsets * stride_xk,
            mask=k_offsets < K,
            other=0.0
        ).to(tl.float32)  # [BLOCK_K], fp32

        # Load W_tile: W[k0:k0+BLOCK_K, :] as [BLOCK_K, N]
        n_offsets = tl.arange(0, N)
        w_tile = tl.load(
            W_ptr + k_offsets[:, None] * stride_wk + n_offsets[None, :] * stride_wn,
            mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N),
            other=0.0
        ).to(tl.float32)  # [BLOCK_K, N], fp32

        # Accumulate: acc += sum over k of x_vec[k] * w_tile[k, :]
        acc += tl.sum(x_vec[:, None] * w_tile, axis=0)  # reduce over k -> [N]

    # Store acc to Y[m, :]
    # n_offsets < N is always true since N is constexpr and loop up to N, but keep mask for safety
    n_offsets = tl.arange(0, N)
    tl.store(Y_ptr + m * stride_ym + n_offsets * stride_yn, acc, mask=True)


@triton.jit
def silu_mul_kernel(
    GateOut_ptr,  # *float32, shape [M, N]
    UpOut_ptr,    # *float32, shape [M, N]
    Y_ptr,        # *float32, shape [M, N]
    M, N,
    stride_gm, stride_gn,
    stride_om, stride_on,
    stride_ym, stride_yn,
):
    # 2D grid over rows and columns
    m = tl.program_id(0)
    n = tl.program_id(1)

    # Bounds check
    if m >= M or n >= N:
        return

    # Load GateOut[m, n] and UpOut[m, n]
    gate = tl.load(GateOut_ptr + m * stride_gm + n * stride_gn)
    up = tl.load(UpOut_ptr + m * stride_om + n * stride_on)

    # Compute SiLU(gate) = gate * sigmoid(gate)
    sig = 1.0 / (1.0 + tl.exp(-gate))
    y = gate * sig * up

    # Store result
    tl.store(Y_ptr + m * stride_ym + n * stride_yn, y)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        grad_output: torch.Tensor,
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        router_logits: torch.Tensor,
        scores: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        score_mask: torch.Tensor,
        shared_expert_gate_weight: torch.Tensor,
        shared_expert_up_weight: torch.Tensor,
        shared_expert_down_weight: torch.Tensor,
        shared_gate_output: torch.Tensor,
        shared_up_output: torch.Tensor,
        shared_activated: torch.Tensor,
    ):
        # Only use hidden_states, shared_expert_gate_weight, shared_expert_up_weight
        # The rest are unused per original forward signature, kept for API compatibility.

        # Ensure CUDA device
        if hidden_states.device.type != 'cuda':
            # Robust fallback: use torch for non-CUDA, avoids runtime errors
            gate_out = hidden_states @ shared_expert_gate_weight.t()  # torch GEMV
            up_out = hidden_states @ shared_expert_up_weight.t()
            activated = torch.nn.functional.silu(gate_out) * up_out
            return activated.to(torch.bfloat16)

        # Make inputs contiguous
        hidden = hidden_states.contiguous()  # [M, K]
        gw = shared_expert_gate_weight.contiguous()  # [K, N]
        upw = shared_expert_up_weight.contiguous()   # [K, N]

        M, K = hidden.shape
        Kgw, N = gw.shape
        assert Kgw == K, f"Weight gate shape mismatch: hidden K={K}, gate_weight K={Kgw}"
        assert upw.shape[1] == N, f"Weight up shape mismatch: N mismatch {upw.shape[1]} vs {N}"

        # Output buffers (float32)
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        activated = torch.empty((M, N), dtype=torch.float32, device=hidden.device)

        # Launch row-wise GEMV kernels for gate_out and up_out
        BLOCK_K = 256
        grid = (M,)

        linear_rowwise_bf16_to_f32[grid](
            hidden, gw, gate_out,
            M, K, N,
            hidden.stride(0), hidden.stride(1),
            gw.stride(0), gw.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        linear_rowwise_bf16_to_f32[grid](
            hidden, upw, up_out,
            M, K, N,
            hidden.stride(0), hidden.stride(1),
            upw.stride(0), upw.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Elementwise SiLU multiply
        grid_elem = (M, N)
        silu_mul_kernel[grid_elem](
            gate_out, up_out, activated,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated.stride(0), activated.stride(1),
            num_warps=4, num_stages=1,
        )

        # Return bfloat16 (cast via dtype constructor)
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
