import torch
import triton
import triton.language as tl


@triton.jit
def linear_rowwise_bf16_to_f32(
    X_ptr,           # *bfloat16, [M, K]
    W_ptr,           # *bfloat16, [K, N]
    Y_ptr,           # *float32,  [M, N]
    M: tl.constexpr, # compile-time M (used for grid; not strictly required)
    K: tl.constexpr, # int
    N: tl.constexpr, # int
    stride_xm,       # int
    stride_xk,       # int
    stride_wk,       # int
    stride_wn,       # int
    stride_ym,       # int
    stride_yn,       # int
    BLOCK_K: tl.constexpr,  # tile size for K (we use 1 here to keep simple and robust)
):
    m = tl.program_id(0)
    if m >= M:
        return
    acc = tl.zeros((N,), dtype=tl.float32)
    # Iterate over K dimension
    for k in range(0, K):
        x_val = tl.load(X_ptr + m * stride_xm + k * stride_xk)  # load scalar x[m, k]
        # load W[k, :] vector of length N
        w_vec = tl.load(W_ptr + k * stride_wk + tl.arange(0, N) * stride_wn, mask=tl.arange(0, N) < N, other=0.0)
        # promote x_val to float32
        x_val_f32 = x_val.to(tl.float32)
        # accumulate
        acc += x_val_f32 * w_vec
    # store acc to Y[m, :]
    tl.store(Y_ptr + m * stride_ym + tl.arange(0, N) * stride_yn, acc, mask=tl.arange(0, N) < N)


@triton.jit
def silu_mul_kernel(
    A_ptr,           # *float32, gate_out [M, N]
    B_ptr,           # *float32, up_out   [M, N]
    C_ptr,           # *float32, activated [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    stride_am,       # int
    stride_an,       # int
    stride_bm,       # int
    stride_bn,       # int
    stride_cm,       # int
    stride_cn,       # int
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if (pid_m >= M) or (pid_n >= N):
        return
    a = tl.load(A_ptr + pid_m * stride_am + pid_n * stride_an)
    b = tl.load(B_ptr + pid_m * stride_bm + pid_n * stride_bn)
    # y = a * sigmoid(a) * b
    sig = 1.0 / (1.0 + tl.exp(-a))
    c = a * sig * b
    tl.store(C_ptr + pid_m * stride_cm + pid_n * stride_cn, c)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # The original forward signature returns (grad_hidden, grad_router_weight, grad_gate, grad_up, grad_down)
        # But our task is to compute and return the forward output shared_activated in Triton.
        # Extract hidden_states, gate_weight, up_weight. In get_inputs, args order is:
        # (grad_output, hidden_states, router_weight, e_score_correction_bias, router_logits, scores, topk_indices, topk_weights, score_mask, shared_expert_gate_weight, shared_expert_up_weight, shared_expert_down_weight, shared_gate_output, shared_up_output, shared_activated)
        # We need hidden_states, gate_weight (router_weight), up_weight (shared_expert_gate_weight), and up_weight (shared_expert_up_weight).
        # However, according to the original, the forward computes with shared_expert_gate_weight and shared_expert_up_weight.
        # So we'll take hidden_states, gate_weight, up_weight from args, ignoring others.
        hidden_states = args[1]  # [M, K]
        gate_weight = args[8]    # [K, N]
        up_weight = args[9]      # [K, N]

        M, K = hidden_states.shape
        K_w, N = gate_weight.shape
        assert K_w == K, "Weight K must match hidden_states K"
        assert up_weight.shape == (K, N), "up_weight must be [K, N]"

        # Ensure contiguous and CUDA
        hidden_states = hidden_states.contiguous()
        gate_weight = gate_weight.contiguous()
        up_weight = up_weight.contiguous()
        device = hidden_states.device

        # Allocate outputs (float32 for computation)
        gate_out = torch.empty((M, N), dtype=torch.float32, device=device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=device)
        activated = torch.empty((M, N), dtype=torch.float32, device=device)

        # Launch row-wise linear kernels: one program per row
        grid = (M,)
        # BLOCK_K is 1 to keep it simple and robust; the loop inside handles all K
        linear_rowwise_bf16_to_f32[grid](
            hidden_states, gate_weight, gate_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_K=1,
            num_warps=1, num_stages=1
        )
        linear_rowwise_bf16_to_f32[grid](
            hidden_states, up_weight, up_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_K=1,
            num_warps=1, num_stages=1
        )

        # Elementwise activation
        silu_mul_kernel[(M, N)](
            gate_out, up_out, activated,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated.stride(0), activated.stride(1),
            num_warps=4, num_stages=1
        )

        # Return bfloat16 as per original get_inputs default output dtype
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
