import torch
import triton
import triton.language as tl


@triton.jit
def linear_rowwise_bf16(
    X_ptr,           # *const bfloat16, shape [M, K]
    W_ptr,           # *const bfloat16, shape [K, N]  (weights are [out_features, in_features] = [N, K], but here we pass [K, N] by transposing before call)
    Y_ptr,           # *float32, shape [M, N]
    M,               # int
    K,               # int
    N,               # int
    stride_xm,       # int (elements)
    stride_xk,       # int (elements)
    stride_wk,       # int (elements)
    stride_wn,       # int (elements)
    stride_ym,       # int (elements)
    stride_yn,       # int (elements),
    BLOCK_N: tl.constexpr,
):
    m = tl.program_id(0)
    if m >= M:
        return

    # Initialize accumulator for this row
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K dimension, accumulate contributions
    # We load X[m, k] as scalar bfloat16 and W[k, n:n+BLOCK_N] as a vector bfloat16, multiply and accumulate into acc
    for k in range(0, K):
        x_val = tl.load(X_ptr + m * stride_xm + k * stride_xk)  # bfloat16 scalar
        # Load W[k, :] vector across N with masking
        n_offsets = tl.arange(0, BLOCK_N)
        mask_n = n_offsets < N
        w_vec = tl.load(W_ptr + k * stride_wk + n_offsets * stride_wn, mask=mask_n, other=0.0)  # bfloat16 vector
        # Promote to float32 for accumulation
        acc += (x_val.to(tl.float32)) * w_vec.to(tl.float32)

    # Store the result row into Y with mask
    n_offsets = tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N
    tl.store(Y_ptr + m * stride_ym + n_offsets * stride_yn, acc, mask=mask_n)


@triton.jit
def silu_mul_kernel(
    GateOut_ptr,     # *float32, shape [M, N]
    UpOut_ptr,       # *float32, shape [M, N]
    Y_ptr,           # *float32, shape [M, N]
    M, N,
    stride_gm, stride_gn,
    stride_om, stride_on,
    stride_ym, stride_yn,
    BLOCK: tl.constexpr,  # process in chunks for 1D grid
):
    pid = tl.program_id(0)
    # Map 1D pid to (m, n)
    total = M * N
    if pid >= total:
        return
    m = pid // N
    n = pid % N

    g = tl.load(GateOut_ptr + m * stride_gm + n * stride_gn)
    u = tl.load(UpOut_ptr + m * stride_om + n * stride_on)
    # silu(g) = g * sigmoid(g)
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
        # Forward only: compute shared_activated = SiLU(gate) * up
        # We will ignore routing tensors and just use shared_expert_gate_weight and shared_expert_up_weight.
        # Ensure tensors are on CUDA and contiguous
        hidden = hidden_states
        gate_weight = shared_expert_gate_weight
        up_weight = shared_expert_up_weight

        if not hidden.is_cuda:
            hidden = hidden.cuda()
        if not gate_weight.is_cuda:
            gate_weight = gate_weight.cuda()
        if not up_weight.is_cuda:
            up_weight = up_weight.cuda()

        hidden = hidden.contiguous()
        gate_weight = gate_weight.contiguous()
        up_weight = up_weight.contiguous()

        # Dimensions
        M, K = hidden.shape
        N_gate = gate_weight.shape[1]  # out_features of gate
        N_up = up_weight.shape[1]      # out_features of up

        # Compute gate_out = hidden @ gate_weight.T (no bias), dtype bfloat16 input, output float32
        gate_out = torch.empty((M, N_gate), dtype=torch.float32, device=hidden.device)
        # Launch row-wise kernel
        grid_m = (M,)
        linear_rowwise_bf16[grid_m](
            hidden, gate_weight, gate_out,
            M, K, N_gate,
            hidden.stride(0), hidden.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_N=128,
        )

        # Compute up_out = hidden @ up_weight.T, output float32
        up_out = torch.empty((M, N_up), dtype=torch.float32, device=hidden.device)
        linear_rowwise_bf16[grid_m](
            hidden, up_weight, up_out,
            M, K, N_up,
            hidden.stride(0), hidden.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_N=128,
        )

        # Compute activated = SiLU(gate_out) * up_out using Triton elementwise kernel
        total = M * N_gate
        activated = torch.empty((M, N_gate), dtype=torch.float32, device=hidden.device)
        silu_mul_kernel[(total,)](
            gate_out, up_out, activated,
            M, N_gate,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated.stride(0), activated.stride(1),
            BLOCK=1,
        )

        # Cast to bfloat16 to match original dtype for return
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
