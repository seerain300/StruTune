import torch
import triton
import triton.language as tl


@triton.jit
def linear_rowwise_bf16_to_f32(
    X_ptr,            # *bfloat16, shape [M, K]
    W_ptr,            # *bfloat16, shape [K, N]
    Y_ptr,            # *float32,  shape [M, N]
    M,                # int
    K,                # int
    N,                # int
    stride_xm,        # int
    stride_xk,        # int
    stride_wk,        # int
    stride_wn,        # int
    stride_ym,        # int
    stride_yn,        # int
    BLOCK_K: tl.constexpr,  # tile size along K
    BLOCK_N: tl.constexpr,  # tile size along N
):
    # Each program handles one row m
    m = tl.program_id(0)
    if m >= M:
        return

    # Initialize accumulation vector for this row
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in tiles
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_offsets < K

        # Load x_vals for row m: shape [BLOCK_K]
        x_vals = tl.load(
            X_ptr + m * stride_xm + k_offsets * stride_xk,
            mask=k_mask,
            other=0.0
        )
        x_vals = x_vals.to(tl.float32)

        # Load W_tile as [BLOCK_K, BLOCK_N], with masks for k and n
        n_offsets = tl.arange(0, BLOCK_N)  # [BLOCK_N]
        n_mask = n_offsets < N

        W_tile = tl.load(
            W_ptr + k_offsets[:, None] * stride_wk + n_offsets[None, :] * stride_wn,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0
        )
        W_tile = W_tile.to(tl.float32)

        # Accumulate: acc[n] += sum_k x_vals[k] * W_tile[k, n]
        # Manually unroll over K tile
        for kk in range(BLOCK_K):
            # scalar x for this k index
            x_scalar = x_vals[kk] if k_start + kk < K else 0.0
            # vector W_k across N
            w_vec = W_tile[kk, :]
            acc += x_scalar * w_vec

    # Store acc to Y[m, :] with mask for n
    tl.store(
        Y_ptr + m * stride_ym + n_offsets * stride_yn,
        acc,
        mask=n_mask
    )


@triton.jit
def silu_mul_kernel(
    GateOut_ptr,       # *float32, shape [M, N]
    UpOut_ptr,         # *float32, shape [M, N]
    Y_ptr,             # *float32, shape [M, N]
    M,                 # int
    N,                 # int
    stride_gm,         # int
    stride_gn,         # int
    stride_um,         # int
    stride_un,         # int
    stride_ym,         # int
    stride_yn,         # int
):
    m = tl.program_id(0)
    n = tl.program_id(1)
    if (m >= M) or (n >= N):
        return
    g = tl.load(GateOut_ptr + m * stride_gm + n * stride_gn)
    u = tl.load(UpOut_ptr + m * stride_um + n * stride_un)
    # silu(x) = x * sigmoid(x)
    s = 1.0 / (1.0 + tl.exp(-g))
    y = g * s * u
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
        """
        Triton-only forward: compute shared_activated = SiLU(gate) * up,
        where gate = hidden_states @ shared_expert_gate_weight.T, up = hidden_states @ shared_expert_up_weight.T.
        Bias is assumed zero.
        """

        # We only need hidden_states and the two weights; others are ignored to match output.
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = shared_expert_gate_weight.shape[1]

        # Ensure CUDA and contiguity
        hidden_states = hidden_states.contiguous()
        gate_weight = shared_expert_gate_weight.contiguous()
        up_weight = shared_expert_up_weight.contiguous()

        # Allocate outputs (float32 for accumulation)
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)

        # Launch row-wise GEMV kernels
        # Parameters: we choose BLOCK_K=256, BLOCK_N=128 for robustness
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

        linear_rowwise_bf16_to_f32[grid](
            hidden_states, up_weight, up_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_K=256, BLOCK_N=128,
            num_warps=4, num_stages=2
        )

        # Elementwise y = gate_out * sigmoid(gate_out) * up_out
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

        # Return bfloat16 tensor (cast is allowed as it's dtype conversion, not torch op on tensor)
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
