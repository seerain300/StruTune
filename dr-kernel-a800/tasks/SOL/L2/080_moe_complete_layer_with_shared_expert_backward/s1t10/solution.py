import torch
import triton
import triton.language as tl


# Triton GEMV: y[b, e] = sum_h hidden[b, h] * W[e, h]
# hidden: [B, H] (bf16), W: [N, H] (bf16), y: [B, N] (f32)
@triton.jit
def gemv_linear_kernel(
    hidden_ptr,   # *bf16, [B, H]
    W_ptr,        # *bf16, [N, H]
    y_ptr,        # *f32,  [B, N]
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    stride_h_b, stride_h_h,
    stride_W_e, stride_W_h,
    stride_y_b, stride_y_e,
    BLOCK_H: tl.constexpr
):
    pid_b = tl.program_id(0)  # batch row
    pid_e = tl.program_id(1)  # expert index
    acc = 0.0
    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        h_vals = tl.load(hidden_ptr + pid_b * stride_h_b + offs_h * stride_h_h, mask=mask_h, other=0.0).to(tl.float32)
        W_vals = tl.load(W_ptr + pid_e * stride_W_e + offs_h * stride_W_h, mask=mask_h, other=0.0).to(tl.float32)
        acc += tl.sum(h_vals * W_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_e * stride_y_e, acc)


# Triton elementwise SiLU: y = x * sigmoid(x), operate on a flat vector (f32)
@triton.jit
def silu_elemwise_kernel(x_ptr, y_ptr, N_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x_f32))
    y = x_f32 * sig
    tl.store(y_ptr + offs, y, mask=mask)


# Triton matmul: C[M, N] = A[M, K] @ B[K, N]
# Compute shared_activated: A=[B, N] (f32), B=[N, H] (bf16), C=[B, H] (f32)
@triton.jit
def matmul_kernel(
    A_ptr,  # *bf16 or *f32, [M, K]
    B_ptr,  # *bf16, [K, N]
    C_ptr,  # *f32,  [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_A_m, stride_A_k,
    stride_B_k, stride_B_n,
    stride_C_m, stride_C_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)  # over rows (batch)
    pid_n = tl.program_id(1)  # over cols (hidden)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_A_m + offs_k[None, :] * stride_A_k,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        ).to(tl.float32)
        b = tl.load(
            B_ptr + offs_k[:, None] * stride_B_k + offs_n[None, :] * stride_B_n,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        ).to(tl.float32)
        acc += tl.dot(a, b)
    tl.store(
        C_ptr + offs_m[:, None] * stride_C_m + offs_n[None, :] * stride_C_n,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # We ignore the first 9 args (grad_output, hidden, router_weight, e_score_correction_bias, router_logits, scores,
        # topk_indices, topk_weights, score_mask). The original forward returns only the shared expert outputs.
        # Extract required tensors: hidden, gate_weight, up_weight, down_weight
        hidden = args[1]  # [B, H], bfloat16
        gate_w = args[9]  # shared_expert_gate_weight, [H, N] = [4096, 1408], bfloat16
        up_w = args[10]   # shared_expert_up_weight,   [H, N] = [4096, 1408], bfloat16
        down_w = args[11] # shared_expert_down_weight, [H, N] = [4096, 1408], bfloat16

        # Ensure contiguous for simple indexing
        hidden_c = hidden.contiguous()
        gate_w_c = gate_w.contiguous()
        up_w_c = up_w.contiguous()
        down_w_c = down_w.contiguous()

        B = hidden_c.shape[0]
        H = hidden_c.shape[1]  # 4096
        N = gate_w_c.shape[1]  # 1408

        # 1) Compute shared_gate_output = F.linear(hidden, gate_w) -> [B, N] (f32)
        shared_gate_out = torch.empty((B, N), dtype=torch.float32, device=hidden_c.device)
        gemv_linear_kernel[(B, N)](
            hidden_c, gate_w_c, shared_gate_out,
            B, H, N,
            hidden_c.stride(0), hidden_c.stride(1),
            gate_w_c.stride(0), gate_w_c.stride(1),
            shared_gate_out.stride(0), shared_gate_out.stride(1),
            BLOCK_H=128,
            num_warps=4, num_stages=2
        )

        # 2) Compute shared_up_output = F.linear(hidden, up_w) -> [B, N] (f32)
        shared_up_out = torch.empty((B, N), dtype=torch.float32, device=hidden_c.device)
        gemv_linear_kernel[(B, N)](
            hidden_c, up_w_c, shared_up_out,
            B, H, N,
            hidden_c.stride(0), hidden_c.stride(1),
            up_w_c.stride(0), up_w_c.stride(1),
            shared_up_out.stride(0), shared_up_out.stride(1),
            BLOCK_H=128,
            num_warps=4, num_stages=2
        )

        # 3) Compute activated = SiLU(gate) * up (elementwise in f32)
        a_vec = torch.empty((B, N), dtype=torch.float32, device=hidden_c.device)
        # Flatten and launch Triton over B*N
        silu_elemwise_kernel[(B * N,)](
            shared_gate_out, a_vec, N_elements=B * N, BLOCK=256, num_warps=4, num_stages=2
        )
        # Multiply by up
        a_vec = a_vec * shared_up_out  # f32

        # 4) Compute shared_activated = F.linear(a_vec, down_w) -> [B, H] (f32), then cast to bfloat16
        shared_activated = torch.empty((B, H), dtype=torch.float32, device=hidden_c.device)
        matmul_kernel[(B, H)](
            a_vec, down_w_c, shared_activated,
            M=B, N=H, K=N,
            stride_A_m=a_vec.stride(0), stride_A_k=a_vec.stride(1),
            stride_B_k=down_w_c.stride(0), stride_B_n=down_w_c.stride(1),
            stride_C_m=shared_activated.stride(0), stride_C_n=shared_activated.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2
        )
        shared_activated_bf16 = shared_activated.to(torch.bfloat16)

        # Return exactly the same three outputs as original forward:
        # (shared_gate_output, shared_up_output, shared_activated), all in bfloat16 to match original
        shared_gate_out_bf16 = shared_gate_out.to(torch.bfloat16)
        shared_up_out_bf16 = shared_up_out.to(torch.bfloat16)
        return shared_gate_out_bf16, shared_up_out_bf16, shared_activated_bf16


def run(*args):
    return ModelNew()(*args)
