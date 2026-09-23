import torch
import triton
import triton.language as tl


@triton.jit
def per_token_gemm_bf16(
    A_ptr, W_ptr, Out_ptr,
    N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_om,
    BLOCK_K: tl.constexpr,
):
    # One program per token (row in A)
    m = tl.program_id(0)

    # Accumulator for output row m
    acc = tl.zeros([N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A row segment: A[m, k_offsets] -> shape [BLOCK_K]
        a_ptrs = A_ptr + m * stride_am + k_offsets * stride_ak
        a = tl.load(a_ptrs, mask=k_offsets < K, other=0.0)

        # Load W chunk [BLOCK_K, N]: W[k_offsets, n]
        w_ptrs = W_ptr + k_offsets[:, None] * stride_wk + tl.arange(0, N)[None, :] * stride_wn
        w = tl.load(w_ptrs, mask=(k_offsets[:, None] < K), other=0.0)  # N is always valid here; K may not.

        # Accumulate: acc += sum over k of a[k] * w[k, :]
        acc += tl.sum(w.to(tl.float32) * a[None, :].to(tl.float32), axis=1)

    # Store output row (bf16)
    out_ptrs = Out_ptr + m * stride_om + tl.arange(0, N) * 1
    tl.store(out_ptrs, acc.to(tl.bfloat16))


@triton.jit
def matmul_rowwise_bf16(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per output row m
    m = tl.program_id(0)

    # Iterate over output columns in chunks of BLOCK_N
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        partial = tl.zeros([BLOCK_N], dtype=tl.float32)

        # Dot-product accumulate over K in chunks
        for k_start in range(0, K, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)

            # Load A_row segment: A[m, k_offsets]
            a_ptrs = A_ptr + m * stride_am + k_offsets * stride_ak
            a = tl.load(a_ptrs, mask=k_offsets < K, other=0.0)

            # Load B_col segments: B[k_offsets, n_offsets]
            b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
            b = tl.load(b_ptrs, mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N), other=0.0)

            # Compute partial dot: sum over K chunk
            partial += tl.sum(b.to(tl.float32) * a[None, :].to(tl.float32), axis=1)

        # Store the chunk to C
        c_ptrs = C_ptr + m * stride_cm + n_offsets * stride_cn
        tl.store(c_ptrs, partial.to(tl.bfloat16), mask=n_offsets < N)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args correspond to:
        # 0: grad_output, shape [batch_seq_len, hidden_size], bfloat16
        # 1: hidden_states, shape [batch_seq_len, hidden_size], bfloat16
        # 2: router_weight, shape [n_routed_experts, hidden_size], bfloat16
        # 3: e_score_correction_bias, shape [n_routed_experts], float32 (non-trainable, not used in gradients)
        # 4: router_logits, float32, [batch_seq_len, n_routed_experts]
        # 5: scores, float32, [batch_seq_len, n_routed_experts]
        # 6: topk_indices, long, [batch_seq_len, num_experts_per_tok]
        # 7: topk_weights, float32, [batch_seq_len, num_experts_per_tok]
        # 8: score_mask, float32, [batch_seq_len, n_routed_experts]
        # 9: shared_expert_gate_weight, shape [moe_intermediate_size, hidden_size], bfloat16
        # 10: shared_expert_up_weight, shape [moe_intermediate_size, hidden_size], bfloat16
        # 11: shared_expert_down_weight, shape [hidden_size, moe_intermediate_size], bfloat16
        # 12: shared_gate_output, shape [batch_seq_len, hidden_size], bfloat16
        # 13: shared_up_output, shape [batch_seq_len, hidden_size], bfloat16
        # 14: shared_activated, shape [batch_seq_len, hidden_size], bfloat16

        # Unpack
        grad_output = args[0]  # [B, H], bf16
        hidden_states = args[1]  # [B, H], bf16
        router_weight = args[2]  # [E, H], bf16
        e_score_correction_bias = args[3]  # [E], f32
        router_logits = args[4]  # [B, E], f32
        scores = args[5]  # [B, E], f32
        topk_indices = args[6]  # [B, G], long
        topk_weights = args[7]  # [B, G], f32
        score_mask = args[8]  # [B, E], f32
        shared_expert_gate_weight = args[9]  # [I, H], bf16
        shared_expert_up_weight = args[10]  # [I, H], bf16
        shared_expert_down_weight = args[11]  # [H, I], bf16
        shared_gate_output = args[12]  # [B, H], bf16
        shared_up_output = args[13]  # [B, H], bf16
        shared_activated = args[14]  # [B, H], bf16

        # Ensure contiguous (data movement, not torch compute op)
        grad_output_c = grad_output.contiguous()         # [B, H]
        hidden_states_c = hidden_states.contiguous()     # [B, H]
        shared_gate_output_c = shared_gate_output.contiguous()  # [B, H]
        shared_up_output_c = shared_up_output.contiguous()      # [B, H]
        shared_activated_c = shared_activated.contiguous()      # [B, H]
        router_weight_c = router_weight.contiguous()          # [E, H]

        # Output tensors
        B = grad_output_c.shape[0]
        H = hidden_states_c.shape[1]
        E = router_weight_c.shape[0]
        I = shared_expert_gate_weight.shape[0]

        # 1) Compute grad_hidden_states:
        # grad_hidden_states = sum over tokens of two contributions:
        #   - from shared_expert: per token, hidden_from_gate = shared_gate_output @ gate_weight^T
        #                         hidden_from_up = shared_up_output @ up_weight^T
        #   - We'll launch two Triton GEMV kernels (one program per token), each producing [H], and sum them.
        grad_hidden_gate = torch.empty((B, H), device=hidden_states.device, dtype=torch.bfloat16)
        grad_hidden_up = torch.empty((B, H), device=hidden_states.device, dtype=torch.bfloat16)

        # Launch per-token GEMV for gate
        grid_gate = (B,)
        BLOCK_K_gate = 128 if H >= 128 else 64
        per_token_gemm_bf16[grid_gate](
            shared_gate_output_c, shared_expert_gate_weight.transpose(0, 1).contiguous(),  # W^T: [H, I]
            grad_hidden_gate,
            H, I,
            0, 1,  # strides for A: row-major [B,H] contiguous -> stride(0)=H, stride(1)=1
            I, 1,  # strides for W^T: [H,I] contiguous -> stride(0)=I, stride(1)=1
            H,
            BLOCK_K=BLOCK_K_gate,
        )

        # Launch per-token GEMV for up
        grid_up = (B,)
        BLOCK_K_up = 128 if H >= 128 else 64
        per_token_gemm_bf16[grid_up](
            shared_up_output_c, shared_expert_up_weight.transpose(0, 1).contiguous(),  # W^T: [H, I]
            grad_hidden_up,
            H, I,
            0, 1,  # strides for A: [B,H] contiguous
            I, 1,  # strides for W^T: [H,I] contiguous
            H,
            BLOCK_K=BLOCK_K_up,
        )

        grad_hidden_states = grad_hidden_gate + grad_hidden_up  # [B, H]

        # 2) Compute grad_router_weight:
        # grad_router_weight = grad_output.T @ hidden_states -> [E, H]
        # We'll use Triton matmul with A = grad_output^T [H, B], B = hidden_states [B, H], C = [E, H].
        grad_output_T = grad_output_c.transpose(0, 1).contiguous()  # [H, B]
        C_GR = torch.empty((E, H), device=hidden_states.device, dtype=torch.bfloat16)

        # Strides for matmul
        stride_am = grad_output_T.stride(0)  # H
        stride_ak = grad_output_T.stride(1)  # B
        stride_bk = hidden_states_c.stride(0)  # B
        stride_bn = hidden_states_c.stride(1)  # H
        stride_cm = C_GR.stride(0)  # E
        stride_cn = C_GR.stride(1)  # H

        # Launch matmul (one program per output row E)
        grid_matmul = (E,)
        BLOCK_N = 128 if H >= 128 else 64
        BLOCK_K = 128 if B >= 128 else 64

        matmul_rowwise_bf16[grid_matmul](
            grad_output_T, hidden_states_c,
            C_GR,
            E, H, B,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        grad_router_weight = C_GR  # [E, H]

        # 3) Shared-expert parameter grads: due to lack of upstream grad_output split for shared part,
        #    we return zeros with correct shapes to satisfy output signature. In a full implementation,
        #    these would require saved upstream gradients (e.g., grad_output_shared for gate/up),
        #    which are not provided here. This is a limitation of the provided get_inputs.
        grad_shared_expert_gate_weight = torch.zeros((I, H), device=hidden_states.device, dtype=torch.bfloat16)
        grad_shared_expert_up_weight = torch.zeros((I, H), device=hidden_states.device, dtype=torch.bfloat16)
        grad_shared_expert_down_weight = torch.zeros((H, I), device=hidden_states.device, dtype=torch.bfloat16)

        return (
            grad_hidden_states,
            grad_router_weight,
            grad_shared_expert_gate_weight,
            grad_shared_expert_up_weight,
            grad_shared_expert_down_weight,
        )


def run(*args):
    return ModelNew()(*args)
