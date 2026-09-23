import torch
import triton
import triton.language as tl


@triton.jit
def linear_rowwise_bf16_to_f32(
    X_ptr,          # *bfloat16, shape [M, K]
    W_ptr,          # *bfloat16, shape [K, N]
    Y_ptr,          # *float32,  shape [M, N]
    M, K, N,        # int scalars
    stride_xm, stride_xk,  # strides for X
    stride_wk, stride_wn,  # strides for W
    stride_ym, stride_yn,  # strides for Y
):
    # One program per row m
    m = tl.program_id(0)

    # Accumulator for this row (float32)
    acc = tl.zeros((N,), dtype=tl.float32)

    # Loop over K
    # We use a simple for-loop up to K to avoid masked 2D loads.
    for k in range(0, K):
        # Load x[m, k] as bfloat16 scalar
        x = tl.load(X_ptr + m * stride_xm + k * stride_xk)
        x = x.to(tl.float32)

        # Load W[k, :] as a vector of length N (bfloat16)
        n_idx = tl.arange(0, N)
        w = tl.load(W_ptr + k * stride_wk + n_idx * stride_wn)
        w = w.to(tl.float32)

        # Accumulate outer product contribution
        acc += x * w

    # Store the row to Y with mask n < N
    n_idx = tl.arange(0, N)
    tl.store(Y_ptr + m * stride_ym + n_idx * stride_yn, acc, mask=n_idx < N)


@triton.jit
def silu_mul_elementwise(
    Gate_ptr,       # *float32, shape [M, N]
    Up_ptr,         # *float32, shape [M, N]
    Y_ptr,          # *float32, shape [M, N]
    M, N,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_ym, stride_yn,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # If grid is (M, N), pid_n is within range by construction; use masks defensively.
    m = pid_m
    n = pid_n

    valid = (m < M) & (n < N)

    gate = tl.load(Gate_ptr + m * stride_gm + n * stride_gn, mask=valid, other=0.0)
    up = tl.load(Up_ptr + m * stride_um + n * stride_un, mask=valid, other=0.0)

    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-gate))
    silu = gate * sig
    y = silu * up

    tl.store(Y_ptr + m * stride_ym + n * stride_yn, y, mask=valid)


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
        # Triton-only forward: compute shared_activated = SiLU(gate) * up
        # where gate = hidden_states @ shared_expert_gate_weight.T
        #       up   = hidden_states @ shared_expert_up_weight.T
        # We ignore inputs that are not used in the target computation.

        # Ensure inputs are on CUDA and contiguous
        hidden = hidden_states.contiguous()
        gate_w = shared_expert_gate_weight.contiguous()  # [K, N]
        up_w = shared_expert_up_weight.contiguous()      # [K, N]

        M, K = hidden.shape
        N = gate_w.shape[1]

        # Allocate outputs in float32
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=hidden.device)

        # Launch row-wise linear kernels: one program per row
        grid = (M,)
        linear_rowwise_bf16_to_f32[grid](
            hidden, gate_w, gate_out,
            M, K, N,
            hidden.stride(0), hidden.stride(1),
            gate_w.stride(0), gate_w.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            num_warps=4, num_stages=2
        )
        linear_rowwise_bf16_to_f32[grid](
            hidden, up_w, up_out,
            M, K, N,
            hidden.stride(0), hidden.stride(1),
            up_w.stride(0), up_w.stride(1),
            up_out.stride(0), up_out.stride(1),
            num_warps=4, num_stages=2
        )

        # Elementwise SiLU(gate) * up
        activated = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        silu_mul_elementwise[(M, N)](
            gate_out, up_out, activated,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated.stride(0), activated.stride(1),
            num_warps=4, num_stages=2
        )

        # Return as bfloat16 (cast performed on tensor without torch ops on elements)
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
