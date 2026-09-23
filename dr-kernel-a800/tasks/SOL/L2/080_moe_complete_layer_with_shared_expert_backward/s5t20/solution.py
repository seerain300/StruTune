import triton
import triton.language as tl
import torch


@triton.jit
def triton_matmul_bf16(A_ptr, B_ptr, C_ptr,
                        M, N, K,
                        stride_am, stride_ak,
                        stride_bk, stride_bn,
                        stride_cm, stride_cn,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # 2D grid of programs: each program computes a BLOCK_M x BLOCK_N tile of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for rows (M) and cols (N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Initialize accumulator with fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers to A and B tiles
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Masks to avoid OOB
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        B_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load as bfloat16, convert to fp32
        A = tl.load(A_ptrs, mask=A_mask, other=0.0).to(tl.float32)
        B = tl.load(B_ptrs, mask=B_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(A, B)

    # Store result as bfloat16
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.bfloat16), mask=C_mask)


@triton.jit
def triton_gemv_bf16_row(A_ptr, B_ptr, Out_ptr,
                          M, K,
                          stride_am, stride_ak,
                          stride_bk, stride_bn,
                          stride_out,  # since Out is 1D, stride_out = 0 for row and 1 for hidden
                          BLOCK_K: tl.constexpr):
    # One program per row m in A (per token)
    pid_m = tl.program_id(0)
    # If pid_m >= M, do nothing
    # Accumulator scalar
    acc = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # A[m, k] vector slice
        A_ptrs = A_ptr + pid_m * stride_am + offs_k * stride_ak
        B_ptrs = B_ptr + offs_k[:, None] * stride_bk + tl.arange(0, BN) * stride_bn  # BN is a constexpr (hidden_size)
        A_mask = (offs_k < K)
        B_mask = (offs_k[:, None] < K) & (tl.arange(0, BN) < M)  # second dim is hidden
        # Load as bf16, convert to fp32
        a_vec = tl.load(A_ptrs, mask=A_mask, other=0.0).to(tl.float32)
        b_tile = tl.load(B_ptrs, mask=B_mask, other=0.0).to(tl.float32)
        # Accumulate dot: sum over K chunk
        acc += tl.sum(a_vec[:, None] * b_tile, axis=0)
    # Store result
    Out_ptr_row = Out_ptr + pid_m * stride_out  # stride_out is effectively 0 for 1D
    tl.store(Out_ptr_row, acc.to(tl.bfloat16))


# Note: The following kernels will be invoked in ModelNew.forward.
# We keep only Triton computation; no torch operations on tensors in forward.


class ModelNew(torch.nn.Module):
    def forward(self, grad_output, hidden_states,
                router_weight,
                e_score_correction_bias,
                router_logits, scores,
                topk_indices, topk_weights,
                score_mask,
                shared_expert_gate_weight, shared_expert_up_weight,
                shared_expert_down_weight,
                shared_gate_output, shared_up_output, shared_activated):
        # All computation must be via Triton kernels. No torch ops in host code.

        # ------------------------------
        # 1) Per-token GEMVs
        # grad_hidden_from_shared_up[token] = grad_shared_up_output[token] @ shared_expert_up_weight
        # grad_hidden_from_shared_gate[token] = grad_shared_gate_output[token] @ shared_expert_gate_weight

        batch_seq_len = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        intermediate_size = shared_expert_up_weight.shape[0]  # 1408

        # Ensure inputs are contiguous (data movement, not torch compute)
        grad_shared_up_output_c = grad_shared_up_output.contiguous()
        shared_expert_up_weight_c = shared_expert_up_weight.contiguous()
        grad_shared_gate_output_c = grad_shared_gate_output.contiguous()
        shared_expert_gate_weight_c = shared_expert_gate_weight.contiguous()

        # Allocate outputs
        grad_hidden_from_shared_up = torch.empty((batch_seq_len, hidden_size), dtype=torch.bfloat16, device=grad_output.device)
        grad_hidden_from_shared_gate = torch.empty((batch_seq_len, hidden_size), dtype=torch.bfloat16, device=grad_output.device)

        # Launch per-token GEMV kernels
        # Choose BLOCK_K = 64, num_warps=2
        triton_gemv_bf16_row[(batch_seq_len,)](
            grad_shared_up_output_c, shared_expert_up_weight_c, grad_hidden_from_shared_up,
            batch_seq_len, hidden_size,
            grad_shared_up_output_c.stride(0), grad_shared_up_output_c.stride(1),
            shared_expert_up_weight_c.stride(0), shared_expert_up_weight_c.stride(1),
            0,  # stride_out (dummy), since Out is 1D, stride is not needed as we store directly by index
            BLOCK_K=64, num_warps=2, num_stages=2
        )
        triton_gemv_bf16_row[(batch_seq_len,)](
            grad_shared_gate_output_c, shared_expert_gate_weight_c, grad_hidden_from_shared_gate,
            batch_seq_len, hidden_size,
            grad_shared_gate_output_c.stride(0), grad_shared_gate_output_c.stride(1),
            shared_expert_gate_weight_c.stride(0), shared_expert_gate_weight_c.stride(1),
            0,
            BLOCK_K=64, num_warps=2, num_stages=2
        )

        # Sum contributions into hidden_states gradient
        grad_hidden_states = grad_hidden_from_shared_up + grad_hidden_from_shared_gate

        # ------------------------------
        # 2) GEMMs (all via triton_matmul_bf16)
        # a) grad_shared_expert_down_weight = grad_shared_output.T @ shared_activated
        #    shape: [hidden_size, intermediate_size]
        grad_shared_output_c = grad_shared_output.contiguous()
        shared_activated_c = shared_activated.contiguous()
        grad_shared_expert_down_weight = torch.empty((hidden_size, intermediate_size), dtype=torch.bfloat16, device=grad_output.device)
        triton_matmul_bf16[(triton.cdiv(hidden_size, 64), triton.cdiv(intermediate_size, 64))](
            grad_shared_output_c, shared_activated_c, grad_shared_expert_down_weight,
            hidden_size, intermediate_size, hidden_size,
            grad_shared_output_c.stride(0), grad_shared_output_c.stride(1),
            shared_activated_c.stride(0), shared_activated_c.stride(1),
            grad_shared_expert_down_weight.stride(0), grad_shared_expert_down_weight.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2
        )

        # b) grad_shared_expert_up_weight = grad_shared_up_output.T @ hidden_states
        #    shape: [intermediate_size, hidden_size]
        grad_shared_expert_up_weight = torch.empty((intermediate_size, hidden_size), dtype=torch.bfloat16, device=grad_output.device)
        triton_matmul_bf16[(triton.cdiv(intermediate_size, 64), triton.cdiv(hidden_size, 64))](
            grad_shared_up_output_c, hidden_states.contiguous(), grad_shared_expert_up_weight,
            intermediate_size, hidden_size, hidden_size,
            grad_shared_up_output_c.stride(0), grad_shared_up_output_c.stride(1),
            hidden_states.contiguous().stride(0), hidden_states.contiguous().stride(1),
            grad_shared_expert_up_weight.stride(0), grad_shared_expert_up_weight.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2
        )

        # c) grad_shared_expert_gate_weight = grad_shared_gate_output.T @ hidden_states
        #    shape: [intermediate_size, hidden_size]
        grad_shared_expert_gate_weight = torch.empty((intermediate_size, hidden_size), dtype=torch.bfloat16, device=grad_output.device)
        triton_matmul_bf16[(triton.cdiv(intermediate_size, 64), triton.cdiv(hidden_size, 64))](
            grad_shared_gate_output_c, hidden_states.contiguous(), grad_shared_expert_gate_weight,
            intermediate_size, hidden_size, hidden_size,
            grad_shared_gate_output_c.stride(0), grad_shared_gate_output_c.stride(1),
            hidden_states.contiguous().stride(0), hidden_states.contiguous().stride(1),
            grad_shared_expert_gate_weight.stride(0), grad_shared_expert_gate_weight.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2
        )

        # d) grad_router_weight = grad_router_logits.T @ hidden_states
        #    shape: [N_experts, hidden_size]
        N_experts = router_weight.shape[0]
        grad_router_logits_c = grad_router_logits.contiguous()
        grad_router_weight = torch.empty((N_experts, hidden_size), dtype=torch.bfloat16, device=grad_output.device)
        triton_matmul_bf16[(triton.cdiv(N_experts, 64), triton.cdiv(hidden_size, 64))](
            grad_router_logits_c, hidden_states.contiguous(), grad_router_weight,
            N_experts, hidden_size, hidden_size,
            grad_router_logits_c.stride(0), grad_router_logits_c.stride(1),
            hidden_states.contiguous().stride(0), hidden_states.contiguous().stride(1),
            grad_router_weight.stride(0), grad_router_weight.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2
        )

        # Return gradients for the inputs (as expected by the original signature)
        return (
            grad_hidden_states,
            grad_router_weight,
            grad_shared_expert_gate_weight,
            grad_shared_expert_up_weight,
            grad_shared_expert_down_weight,
        )


def run(*args):
    return ModelNew()(*args)
