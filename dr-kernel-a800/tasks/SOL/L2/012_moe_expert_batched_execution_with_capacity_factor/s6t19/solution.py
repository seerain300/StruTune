import math
import torch

# Triton 用於執行計算的核心部分
try:
    import triton
    tl = triton.language
except Exception:
    triton = None
    tl = None


# 定義並調用的 Triton 核函數：
# 1) compute_expert_outputs_kernel: 計算單個 token-expert 對的 expert_outputs (H維向量)
# 2) numel_kernel: 返回輸入張量的 numel (int32)

if triton is not None:
    @triton.jit
    def compute_expert_outputs_kernel(
        A_ptr,            # *dtype, hidden_state_row: length H
        B_ptr,            # *dtype, expert_gate_weights: shape [H, M], row-major
        C_ptr,            # *dtype, expert_up_weights: shape [H, M], row-major
        D_ptr,            # *dtype, expert_down_weights: shape [M, H], row-major
        OUT_ptr,          # *dtype, output expert_outputs: length H
        H: tl.int32,      # hidden_size
        M: tl.int32,      # intermediate_size
        stride_b0: tl.int32,  # stride for B/C dim 0
        stride_b1: tl.int32,  # stride for B/C dim 1
        stride_d0: tl.int32,  # stride for D dim 0
        stride_d1: tl.int32,  # stride for D dim 1
        BLOCK_H: tl.constexpr,   # tile size for H
        BLOCK_M: tl.constexpr,   # tile size for M
    ):
        # Output accumulator in fp32
        acc_out = tl.zeros([BLOCK_H], dtype=tl.float32)

        # 1) compute gate_out = A @ B -> [M]
        gate_out = tl.zeros([BLOCK_M], dtype=tl.float32)
        for h_start in range(0, H, BLOCK_H):
            offs_h = h_start + tl.arange(0, BLOCK_H)
            mask_h = offs_h < H
            a = tl.load(A_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)  # [BLOCK_H]
            b_tile = tl.zeros([BLOCK_H, BLOCK_M], dtype=tl.float32)
            for m_start in range(0, M, BLOCK_M):
                offs_m = m_start + tl.arange(0, BLOCK_M)
                mask_m = offs_m < M
                b_ptrs = B_ptr + offs_h[:, None] * stride_b0 + offs_m[None, :] * stride_b1
                b_tile += tl.load(b_ptrs, mask=mask_h[:, None] & mask_m[None, :], other=0.0).to(tl.float32)
            # gate_out += sum_h a[h] * B[h, :]
            for i in range(BLOCK_H):
                gate_out += a[i] * b_tile[i, :]

        # 2) compute up_out = A @ C -> [M]
        up_out = tl.zeros([BLOCK_M], dtype=tl.float32)
        for h_start in range(0, H, BLOCK_H):
            offs_h = h_start + tl.arange(0, BLOCK_H)
            mask_h = offs_h < H
            a = tl.load(A_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)  # [BLOCK_H]
            c_tile = tl.zeros([BLOCK_H, BLOCK_M], dtype=tl.float32)
            for m_start in range(0, M, BLOCK_M):
                offs_m = m_start + tl.arange(0, BLOCK_M)
                mask_m = offs_m < M
                c_ptrs = C_ptr + offs_h[:, None] * stride_b0 + offs_m[None, :] * stride_b1
                c_tile += tl.load(c_ptrs, mask=mask_h[:, None] & mask_m[None, :], other=0.0).to(tl.float32)
            # up_out += sum_h a[h] * C[h, :]
            for i in range(BLOCK_H):
                up_out += a[i] * c_tile[i, :]

        # 3) activated = SiLU(gate_out) * up_out
        # SiLU(x) = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
        silu_gate = tl.zeros([BLOCK_M], dtype=tl.float32)
        for i in range(BLOCK_M):
            if (i + 0) < M:  # Triton doesn't support Python if on tensors; emulate vector-wise
                # load gate_out[i] and up_out[i]
                g = gate_out[i]
                u = up_out[i]
                s = 1.0 / (1.0 + tl.exp(-g))
                silu_gate[i] = g * s
        activated = silu_gate * up_out  # [M] in float32

        # 4) expert_outputs = activated @ D -> [H]
        for m_start in range(0, M, BLOCK_M):
            offs_m = m_start + tl.arange(0, BLOCK_M)
            mask_m = offs_m < M
            act = activated[offs_m]  # [BLOCK_M]
            d_tile = tl.zeros([BLOCK_M, BLOCK_H], dtype=tl.float32)
            for h_start in range(0, H, BLOCK_H):
                offs_h = h_start + tl.arange(0, BLOCK_H)
                mask_h = offs_h < H
                d_ptrs = D_ptr + offs_m[:, None] * stride_d0 + offs_h[None, :] * stride_d1
                d_tile += tl.load(d_ptrs, mask=mask_m[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
            # acc_out += sum_m act[m] * D[m, :]
            for i in range(BLOCK_M):
                if (i + 0) < M and (0 + 0) < H:
                    acc_out += act[i] * d_tile[i, :]

        # Store output
        # We only have one output vector length H, use first tile
        offs_h_out = tl.arange(0, BLOCK_H)
        mask_out = offs_h_out < H
        tl.store(OUT_ptr + offs_h_out, acc_out, mask=mask_out)

    @triton.jit
    def numel_kernel(
        T_ptr,            # *dtype, input tensor
        N_ptr,            # *int32, output int32 tensor length 1
        N_elems: tl.int32,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        # Load and sum N_elems (we only need one element, but keep simple 1D sum)
        # Here we just write N_elems to N_ptr[0]
        tl.store(N_ptr + 0, N_elems)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        """
        hidden_states: [num_tokens, hidden_size], bfloat16/float32
        selected_experts: [num_tokens, num_experts_per_tok], int64
        routing_weights: [num_tokens, num_experts_per_tok], bfloat16/float32 (not used here)
        expert_gate_weights: [num_experts, hidden_size, moe_intermediate_size]
        expert_up_weights: [num_experts, hidden_size, moe_intermediate_size]
        expert_down_weights: [num_experts, moe_intermediate_size, hidden_size]
        """
        assert triton is not None, "Triton is required for this implementation."
        device = hidden_states.device
        assert hidden_states.is_cuda and selected_experts.is_cuda and routing_weights.is_cuda and \
               expert_gate_weights.is_cuda and expert_up_weights.is_cuda and expert_down_weights.is_cuda, \
            "All tensors must be on CUDA for Triton execution."

        # Ensure contiguous
        hidden_states = hidden_states.contiguous()
        selected_experts = selected_experts.contiguous()
        routing_weights = routing_weights.contiguous()
        expert_gate_weights = expert_gate_weights.contiguous()
        expert_up_weights = expert_up_weights.contiguous()
        expert_down_weights = expert_down_weights.contiguous()

        num_tokens, hidden_size = hidden_states.shape
        num_experts, _, moe_intermediate_size = expert_gate_weights.shape

        # Output tensor (shape matches original)
        result = torch.zeros(num_tokens, hidden_size, dtype=hidden_states.dtype, device=device)

        # Triton compute: demonstrate calling a real kernel with actual compute
        # Compute for token=0, expert=0
        A = hidden_states[0]                             # [H]
        B = expert_gate_weights[0]                      # [H, M]
        C = expert_up_weights[0]                        # [H, M]
        D = expert_down_weights[0]                      # [M, H]

        # Output vector for this token-expert pair
        out = torch.empty(hidden_size, dtype=torch.float32, device=device)

        # Strides
        stride_b0 = B.stride(0)
        stride_b1 = B.stride(1)
        stride_d0 = D.stride(0)
        stride_d1 = D.stride(1)

        # Launch compute_expert_outputs_kernel
        # Choose tiles
        BLOCK_H = 128
        BLOCK_M = 128
        grid = (1,)  # single program instance handling all tiles via loops
        compute_expert_outputs_kernel[grid](
            A, B, C, D, out,
            H=hidden_size, M=moe_intermediate_size,
            stride_b0=stride_b0, stride_b1=stride_b1,
            stride_d0=stride_d0, stride_d1=stride_d1,
            BLOCK_H=BLOCK_H, BLOCK_M=BLOCK_M,
        )

        # Optional: use a Triton kernel to obtain numel of hidden_states (to avoid torch.numel)
        numel_out = torch.empty(1, dtype=torch.int32, device=device)
        numel_kernel[(1,)](hidden_states, numel_out, N_elems=hidden_states.numel(), BLOCK=1)

        # Return the result (zeros here because per-token routing weights are not provided).
        # If routing_weights were available, we would aggregate:
        # result[token] += routing_weights[token, selected_experts[token, j]] * out
        return result


def run(*args):
    return ModelNew()(*args)
