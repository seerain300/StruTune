import triton
import triton.language as tl


@triton.jit
def triton_matmul_bf16(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid: each program handles a BLOCK_M x BLOCK_N tile of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Iterate over K in chunks
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


@triton.jit
def triton_gemv_bf16(
    A_ptr, x_ptr, y_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_xk,
    stride_ym,
    BLOCK_K: tl.constexpr,
):
    # One program per row (token)
    pid = tl.program_id(0)
    offs_m = pid
    acc = tl.zeros([N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m * stride_am) + (offs_k * stride_ak)
        x_ptrs = x_ptr + offs_k * stride_xk
        a = tl.load(a_ptrs, mask=offs_k < K, other=0.0).to(tl.float32)
        x = tl.load(x_ptrs, mask=offs_k < K, other=0.0).to(tl.float32)
        acc += tl.sum(a * x, axis=0)

    y_ptrs = y_ptr + offs_m * stride_ym
    tl.store(y_ptrs, acc.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self,
        grad_output,             # [B, H] bfloat16
        hidden_states,           # [B, H] bfloat16
        router_weight,           # [E, H] bfloat16
        e_score_correction_bias, # [E] float32
        router_logits,           # [B, E] float32
        scores,                  # [B, E] float32
        topk_indices,            # [B, G] int64
        topk_weights,            # [B, G] float32
        score_mask,              # [B, E] float32
        shared_expert_gate_weight,   # [M, H] bfloat16
        shared_expert_up_weight,     # [M, H] bfloat16
        shared_expert_down_weight,   # [H, M] bfloat16
        shared_gate_output,          # [B, M] bfloat16
        shared_up_output,            # [B, M] bfloat16
        shared_activated              # [B, M] bfloat16
    ):
        # All heavy computation must be performed by Triton kernels. No torch ops in forward.

        B = grad_output.shape[0]
        H = grad_output.shape[1]
        E = router_weight.shape[0]
        M = shared_expert_gate_weight.shape[0]  # intermediate_size (e.g., 1408)
        K = shared_expert_gate_weight.shape[1]  # hidden_size (e.g., 4096)

        # Constants for tiling (tuneable, but good defaults)
        BM = 64
        BN = 128
        BK = 64

        # 1) grad_hidden_from_shared_up: per-token GEMV
        grad_hidden_from_shared_up = torch.empty((B, H), dtype=torch.bfloat16, device=grad_output.device)
        grid_gemv = (B,)
        for t in range(B):
            # A = shared_up_output[t] shape [M], x = shared_expert_up_weight shape [M, H]
            A_vec = shared_up_output[t]  # [M], ensure pointer-compatible
            B_mat = shared_expert_up_weight  # [M, H]
            y_vec = grad_hidden_from_shared_up[t]  # [H]
            triton_gemv_bf16[grid_gemv](
                A_vec, B_mat, y_vec,
                M, H, M,
                A_vec.stride(0), A_vec.stride(0),
                B_mat.stride(1),
                y_vec.stride(0),
                BLOCK_K=BK,
                num_warps=4,
            )

        # 2) grad_hidden_from_shared_gate: per-token GEMV
        grad_hidden_from_shared_gate = torch.empty((B, H), dtype=torch.bfloat16, device=grad_output.device)
        grid_gemv2 = (B,)
        for t in range(B):
            A_vec2 = shared_gate_output[t]  # [M]
            B_mat2 = shared_expert_gate_weight  # [M, H]
            y_vec2 = grad_hidden_from_shared_gate[t]  # [H]
            triton_gemv_bf16[grid_gemv2](
                A_vec2, B_mat2, y_vec2,
                M, H, M,
                A_vec2.stride(0), A_vec2.stride(0),
                B_mat2.stride(1),
                y_vec2.stride(0),
                BLOCK_K=BK,
                num_warps=4,
            )

        # 3) grad_shared_expert_down_weight = grad_shared_output.T @ shared_activated
        # A = grad_shared_output.T shape [H, H], B = shared_activated shape [H, M]
        grad_shared_output_T = grad_output.transpose(0, 1).contiguous()  # [H, B], then use transpose(0,1) view; to pass strides, we can avoid contig with: grad_output.view(H, B)
        # Triton expects contiguous? We'll use strides: grad_output_T = grad_output with strides [H, B]
        grad_shared_expert_down_weight = torch.empty((H, M), dtype=torch.bfloat16, device=grad_output.device)
        grid_gemm = (triton.cdiv(H, BM), triton.cdiv(M, BN))
        triton_matmul_bf16[grid_gemm](
            grad_output_T, shared_activated, grad_shared_expert_down_weight,
            H, M, H,
            grad_output.stride(0), grad_output.stride(1),
            shared_activated.stride(0), shared_activated.stride(1),
            grad_shared_expert_down_weight.stride(0), grad_shared_expert_down_weight.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            num_warps=4,
        )

        # 4) grad_shared_expert_up_weight = grad_shared_up_output.T @ hidden_states
        # A = grad_shared_up_output.T shape [M, B], B = hidden_states shape [B, H]
        grad_shared_up_output_T = shared_up_output.transpose(0, 1)  # [M, B]
        grad_shared_expert_up_weight = torch.empty((M, H), dtype=torch.bfloat16, device=hidden_states.device)
        grid_gemm2 = (triton.cdiv(M, BM), triton.cdiv(H, BN))
        triton_matmul_bf16[grid_gemm2](
            grad_shared_up_output_T, hidden_states, grad_shared_expert_up_weight,
            M, H, B,
            grad_shared_up_output_T.stride(0), grad_shared_up_output_T.stride(1),
            hidden_states.stride(0), hidden_states.stride(1),
            grad_shared_expert_up_weight.stride(0), grad_shared_expert_up_weight.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            num_warps=4,
        )

        # 5) grad_shared_expert_gate_weight = grad_shared_gate_output.T @ hidden_states
        grad_shared_gate_output_T = shared_gate_output.transpose(0, 1)  # [M, B]
        grad_shared_expert_gate_weight = torch.empty((M, H), dtype=torch.bfloat16, device=hidden_states.device)
        grid_gemm3 = (triton.cdiv(M, BM), triton.cdiv(H, BN))
        triton_matmul_bf16[grid_gemm3](
            grad_shared_gate_output_T, hidden_states, grad_shared_expert_gate_weight,
            M, H, B,
            grad_shared_gate_output_T.stride(0), grad_shared_gate_output_T.stride(1),
            hidden_states.stride(0), hidden_states.stride(1),
            grad_shared_expert_gate_weight.stride(0), grad_shared_expert_gate_weight.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            num_warps=4,
        )

        # 6) grad_router_weight = grad_router_logits.T @ hidden_states
        # A = grad_router_logits.T shape [E, B], B = hidden_states shape [B, H]
        grad_router_logits_T = router_logits.transpose(0, 1)  # [E, B]
        grad_router_weight = torch.empty((E, H), dtype=torch.bfloat16, device=hidden_states.device)
        grid_gemm4 = (triton.cdiv(E, BM), triton.cdiv(H, BN))
        triton_matmul_bf16[grid_gemm4](
            grad_router_logits_T, hidden_states, grad_router_weight,
            E, H, B,
            grad_router_logits_T.stride(0), grad_router_logits_T.stride(1),
            hidden_states.stride(0), hidden_states.stride(1),
            grad_router_weight.stride(0), grad_router_weight.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            num_warps=4,
        )

        # Return gradients corresponding to inputs of original run:
        # The original run returns (grad_hidden_states, grad_router_weight, grad_shared_expert_gate_weight, grad_shared_expert_up_weight, grad_shared_expert_down_weight)
        # We need to combine the per-token contributions for grad_hidden_states:
        grad_hidden_states = grad_hidden_from_shared_up + grad_hidden_from_shared_gate

        return (
            grad_hidden_states,
            grad_router_weight,
            grad_shared_expert_gate_weight,
            grad_shared_expert_up_weight,
            grad_shared_expert_down_weight,
        )


def run(*args):
    return ModelNew()(*args)
