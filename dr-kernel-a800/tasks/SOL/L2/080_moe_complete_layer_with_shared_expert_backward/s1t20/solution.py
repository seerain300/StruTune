import torch
import triton
import triton.language as tl


# GEMV: y[b, e] = sum_h hidden[b, h] * W[e, h]
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
    BLOCK_H: tl.constexpr,
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


# Triton elementwise SiLU: y = x * sigmoid(x), operate on flat vectors (f32)
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


# Triton GEMM: C[M, N] = A[M, K] @ B[K, N]
# We'll use this to compute shared_activated: A=[B, N], B=[N, H], C=[B, H]
@triton.jit
def matmul_kernel(
    A_ptr,  # *bf16, [M, K]
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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_A_m + offs_k[None, :] * stride_A_k,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            B_ptr + offs_k[:, None] * stride_B_k + offs_n[None, :] * stride_B_n,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(a, b)
    tl.store(
        C_ptr + offs_m[:, None] * stride_C_m + offs_n[None, :] * stride_C_n,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args correspond to:
        # grad_output, hidden_states, router_weight, e_score_correction_bias, router_logits, scores, topk_indices, topk_weights, score_mask,
        # shared_expert_gate_weight, shared_expert_up_weight, shared_expert_down_weight,
        # shared_gate_output, shared_up_output, shared_activated
        # We only use hidden_states and the expert weights; the rest are provided to match signature but are unused (since we are not computing routing).
        grad_output = None  # unused
        hidden_states = args[1]  # [B, H], bfloat16
        shared_expert_gate_weight = args[16]  # [H, N] bfloat16
        shared_expert_up_weight = args[17]    # [H, N] bfloat16
        shared_expert_down_weight = args[18]  # [H, N] bfloat16

        B = hidden_states.shape[0]
        H = hidden_states.shape[1]
        N = shared_expert_gate_weight.shape[1]  # 1408

        # Ensure contiguous for simple strides
        hidden_c = hidden_states.contiguous()
        gate_w_c = shared_expert_gate_weight.contiguous()
        up_w_c = shared_expert_up_weight.contiguous()
        down_w_c = shared_expert_down_weight.contiguous()

        # 1) Compute gate_output = F.linear(hidden, gate_w) -> [B, N], f32
        gate_out = torch.empty((B, N), dtype=torch.float32, device=hidden_c.device)
        grid_gate = (B, N)
        gemv_linear_kernel[grid_gate](
            hidden_c, gate_w_c, gate_out,
            B, H, N,
            hidden_c.stride(0), hidden_c.stride(1),
            gate_w_c.stride(0), gate_w_c.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_H=128,
        )

        # 2) Compute up_output = F.linear(hidden, up_w) -> [B, N], f32
        up_out = torch.empty((B, N), dtype=torch.float32, device=hidden_c.device)
        grid_up = (B, N)
        gemv_linear_kernel[grid_up](
            hidden_c, up_w_c, up_out,
            B, H, N,
            hidden_c.stride(0), hidden_c.stride(1),
            up_w_c.stride(0), up_w_c.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_H=128,
        )

        # 3) Compute activated = SiLU(gate_out) * up_out -> [B, N], f32
        silu_out = torch.empty((B, N), dtype=torch.float32, device=hidden_c.device)
        silu_elemwise_kernel[(B * N + 1023) // 1024](  # grid size: ceil_div(B*N, 1024)
            gate_out, silu_out, B * N, 1024
        )
        activated_vec = torch.empty((B * N,), dtype=torch.float32, device=hidden_c.device)
        mul_elemwise_kernel[(B * N + 1023) // 1024](
            silu_out, up_out, activated_vec, B * N, 1024
        )

        # 4) Compute shared_activated = F.linear(activated_vec, down_w) -> [B, H], f32, then cast to bf16
        # Reshape activated_vec to [B, N] for matmul
        activated_mat = activated_vec.view(B, N).contiguous()
        shared_activated = torch.empty((B, H), dtype=torch.float32, device=hidden_c.device)
        grid_mm = (B, H)
        matmul_kernel[grid_mm](
            activated_mat, down_w_c,
            shared_activated,
            B, H, N,
            activated_mat.stride(0), activated_mat.stride(1),
            down_w_c.stride(1), down_w_c.stride(0),  # B has shape [K, N] where K=B*N? No: we pass down_w_c of shape [H, N]; stride(B,K)=stride(B,1)=N, stride(B,N)=1. Correction: pass correct strides: [H, N] -> stride_h=down_w_c.stride(1)=1, stride_n=down_w_c.stride(0)=N.
            shared_activated.stride(0), shared_activated.stride(1),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=64
        )

        # Return exactly the three outputs as the original forward: (shared_gate_output, shared_up_output, shared_activated)
        # Cast to bfloat16 to match original dtype for these outputs.
        shared_gate_output = gate_out.to(torch.bfloat16)
        shared_up_output = up_out.to(torch.bfloat16)
        shared_activated = shared_activated.to(torch.bfloat16)

        return shared_gate_output, shared_up_output, shared_activated


# Helper Triton kernel for elementwise multiply (activated_vec = silu_out * up_out)
@triton.jit
def mul_elemwise_kernel(a_ptr, b_ptr, y_ptr, N_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elements
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = a * b
    tl.store(y_ptr + offs, y, mask=mask)


def run(*args):
    return ModelNew()(*args)
