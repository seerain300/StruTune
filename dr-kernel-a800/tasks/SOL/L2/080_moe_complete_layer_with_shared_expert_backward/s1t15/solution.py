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
# Compute shared_activated: A=[B, N] (pre-activated), B=[N, H], C=[B, H]
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
        a = tl.load(A_ptr + offs_m[:, None] * stride_A_m + offs_k[None, :] * stride_A_k,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                    other=0.0).to(tl.float32)
        b = tl.load(B_ptr + offs_k[:, None] * stride_B_k + offs_n[None, :] * stride_B_n,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0).to(tl.float32)
        acc += tl.dot(a, b)
    tl.store(C_ptr + offs_m[:, None] * stride_C_m + offs_n[None, :] * stride_C_n,
             acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # The original Model.forward returns:
        # (shared_gate_output, shared_up_output, shared_activated)
        # We compute and return these using Triton kernels; do not use any torch ops in forward.

        # Inputs come from the evaluator; positions correspond to original signature.
        # hidden_states is at index -3
        hidden = args[-3]  # [B, H], bfloat16

        # Weights:
        shared_expert_gate_weight = args[-7]  # [H, N] = [4096, 1408], bfloat16
        shared_expert_up_weight    = args[-6]  # [H, N] = [4096, 1408], bfloat16
        shared_expert_down_weight  = args[-5]  # [H, N] = [4096, 1408], bfloat16

        # Ensure contiguity for simple indexing
        hidden = hidden.contiguous()
        gate_w = shared_expert_gate_weight.contiguous()  # [H, N]
        up_w   = shared_expert_up_weight.contiguous()    # [H, N]
        down_w = shared_expert_down_weight.contiguous()  # [H, N]

        B, H = hidden.shape
        N = gate_w.shape[1]  # 1408 in the provided setup

        # 1) Compute shared_gate_output = hidden @ gate_w^T => [B, N] in f32
        shared_gate = torch.empty((B, N), dtype=torch.float32, device=hidden.device)
        grid_gate = (B, N)
        gemv_linear_kernel[grid_gate](
            hidden, gate_w, shared_gate,
            B, H, N,
            hidden.stride(0), hidden.stride(1),
            gate_w.stride(0), gate_w.stride(1),
            shared_gate.stride(0), shared_gate.stride(1),
            BLOCK_H=128
        )

        # 2) Compute shared_up_output = hidden @ up_w^T => [B, N] in f32
        shared_up = torch.empty((B, N), dtype=torch.float32, device=hidden.device)
        grid_up = (B, N)
        gemv_linear_kernel[grid_up](
            hidden, up_w, shared_up,
            B, H, N,
            hidden.stride(0), hidden.stride(1),
            up_w.stride(0), up_w.stride(1),
            shared_up.stride(0), shared_up.stride(1),
            BLOCK_H=128
        )

        # 3) Compute shared_activated = silu(shared_gate) * shared_up, then down projection
        #    activated_flat[b, t] = silu(shared_gate[b, t]) * shared_up[b, t] => [B*N] in f32
        BxN = B * N
        gate_flat = shared_gate.view(BxN).contiguous()
        up_flat = shared_up.view(BxN).contiguous()
        activated_flat = torch.empty(BxN, dtype=torch.float32, device=hidden.device)

        # SiLU elementwise: activated_flat = silu(gate_flat)
        silu_elemwise_kernel[(BxN + 1023) // 1024,](gate_flat, activated_flat, BxN, BLOCK=1024)

        # Elementwise multiply: activated_flat = activated_flat * up_flat
        # Implement via Triton (launch grid over blocks)
        BLOCK_MUL = 1024
        num_progs = (BxN + BLOCK_MUL - 1) // BLOCK_MUL
        # Note: Triton doesn't support arbitrary Python loops in kernels here; we launch per-block.
        for i in range(0, num_progs):
            offs = i * BLOCK_MUL + tl.arange(0, BLOCK_MUL)
            mask = offs < BxN
            a = tl.load(activated_flat + offs, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(up_flat + offs, mask=mask, other=0.0).to(tl.float32)
            y = a * b
            tl.store(activated_flat + offs, y, mask=mask)

        # 4) Down projection: shared_activated = activated_flat @ down_w^T => [B, H] in f32
        activated_mat = activated_flat.view(B, N).contiguous()  # [B, N]
        down_trans = down_w.transpose(0, 1).contiguous()       # [N, H], bfloat16
        shared_activated = torch.empty((B, H), dtype=torch.float32, device=hidden.device)
        grid_down = (B, H)
        matmul_kernel[grid_down](
            activated_mat, down_trans, shared_activated,
            B, H, N,
            activated_mat.stride(0), activated_mat.stride(1),
            down_trans.stride(0), down_trans.stride(1),
            shared_activated.stride(0), shared_activated.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
        )

        # Cast outputs to bfloat16 to match original forward
        shared_gate_output = shared_gate.to(torch.bfloat16)  # [B, N]
        shared_up_output   = shared_up.to(torch.bfloat16)    # [B, N]
        shared_activated   = shared_activated.to(torch.bfloat16)  # [B, H]

        # Return exactly the three outputs as original forward
        return shared_gate_output, shared_up_output, shared_activated


def run(*args):
    return ModelNew()(*args)
