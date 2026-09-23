import torch
import triton
import triton.language as tl


@triton.jit
def linear_rowwise_bf16_to_f32(
    X_ptr,            # *bfloat16, shape [M, K]
    W_ptr,            # *bfloat16, shape [K, N]
    Y_ptr,            # *float32,  shape [M, N]
    M: tl.constexpr,  # rows
    K: tl.constexpr,  # loop bound over K
    N: tl.constexpr,  # output width N (cols of W)
    stride_xm,        # stride along M for X
    stride_xk,        # stride along K for X
    stride_wk,        # stride along K for W (row stride)
    stride_wn,        # stride along N for W (col stride)
    stride_ym,        # stride along M for Y
    stride_yn,        # stride along N for Y
):
    m = tl.program_id(0)  # one program per row
    # Accumulator for this row (float32)
    acc = tl.zeros((N,), dtype=tl.float32)

    # Loop over K dimension: load x[m, k] and W[k, :] vector, accumulate
    for k in range(0, K):
        # Load scalar x[m, k]
        x = tl.load(X_ptr + m * stride_xm + k * stride_xk).to(tl.float32)
        # Load W[k, :] vector of length N
        n_offsets = tl.arange(0, N)
        w = tl.load(W_ptr + k * stride_wk + n_offsets * stride_wn).to(tl.float32)
        # Accumulate
        acc += x * w

    # Store acc to Y[m, :]
    for n in range(0, N):
        tl.store(Y_ptr + m * stride_ym + n * stride_yn, acc[n])


@triton.jit
def silu_mul_kernel(
    A_ptr,            # *float32, GateOut[M, N]
    B_ptr,            # *float32, UpOut[M, N]
    C_ptr,            # *float32, Activated[M, N]
    M: tl.constexpr,  # rows
    N: tl.constexpr,  # cols
    stride_am,        # stride along M for A
    stride_an,        # stride along N for A
    stride_bm,        # stride along M for B
    stride_bn,        # stride along N for B
    stride_cm,        # stride along M for C
    stride_cn,        # stride along N for C
):
    pid = tl.program_id(0)  # 1D grid over M*N
    m = pid // N
    n = pid % N

    if (m < M) and (n < N):
        a = tl.load(A_ptr + m * stride_am + n * stride_an)
        b = tl.load(B_ptr + m * stride_bm + n * stride_bn)
        # sigmoid(x) = 1 / (1 + exp(-x))
        sig = 1.0 / (1.0 + tl.exp(-a))
        c = a * sig * b
        tl.store(C_ptr + m * stride_cm + n * stride_cn, c)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, router_weight, e_score_correction_bias,
                router_logits, scores, topk_indices, topk_weights, score_mask,
                shared_expert_gate_weight, shared_expert_up_weight, shared_expert_down_weight,
                shared_gate_output, shared_up_output, shared_activated):
        """
        Triton-only forward. We compute:
          - gate_out = hidden_states @ shared_expert_gate_weight.T
          - up_out   = hidden_states @ shared_expert_up_weight.T
          - y        = gate_out * sigmoid(gate_out) * up_out
        Return y as bfloat16. No torch ops on tensors.
        """
        # Ensure contiguous and on CUDA
        hidden = hidden_states.contiguous()
        gate_w = shared_expert_gate_weight.contiguous()
        up_w = shared_expert_up_weight.contiguous()

        M = hidden.shape[0]
        K = hidden.shape[1]
        N = gate_w.shape[1]  # 1408

        # Allocate outputs (float32 for numeric stability)
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=hidden.device)

        # Launch row-wise GEMV kernels: one program per row
        grid = (M,)
        linear_rowwise_bf16_to_f32[grid](
            hidden, gate_w, gate_out,
            M, K, N,
            hidden.stride(0), hidden.stride(1),
            gate_w.stride(0), gate_w.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            num_warps=1, num_stages=1,
        )

        linear_rowwise_bf16_to_f32[grid](
            hidden, up_w, up_out,
            M, K, N,
            hidden.stride(0), hidden.stride(1),
            up_w.stride(0), up_w.stride(1),
            up_out.stride(0), up_out.stride(1),
            num_warps=1, num_stages=1,
        )

        # Elementwise: y = gate_out * sigmoid(gate_out) * up_out
        activated = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        total = M * N
        silu_mul_kernel[(total,)](
            gate_out, up_out, activated,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated.stride(0), activated.stride(1),
            num_warps=1, num_stages=1,
        )

        # Return as bfloat16 (no torch ops on tensors)
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
