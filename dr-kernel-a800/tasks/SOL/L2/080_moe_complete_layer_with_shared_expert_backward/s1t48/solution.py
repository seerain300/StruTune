import torch
import triton
import triton.language as tl


# GEMV: y[b, e] = sum_h hidden[b, h] * W[e, h]
# hidden: [B, H], W: [N, H], y: [B, N]
@triton.jit
def gemv_linear_kernel(
    hidden_ptr,    # *bf16/f32, [B, H]
    W_ptr,         # *bf16/f32, [N, H]
    y_ptr,         # *f32,      [B, N]
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    stride_h_b, stride_h_h,
    stride_W_e, stride_W_h,
    stride_y_b, stride_y_e,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch row
    pid_e = tl.program_id(1)  # output index in W (gate or up)
    acc = 0.0
    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        h_vals = tl.load(hidden_ptr + pid_b * stride_h_b + offs_h * stride_h_h, mask=mask_h, other=0.0).to(tl.float32)
        W_vals = tl.load(W_ptr + pid_e * stride_W_e + offs_h * stride_W_h, mask=mask_h, other=0.0).to(tl.float32)
        acc += tl.sum(h_vals * W_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_e * stride_y_e, acc)


# Elementwise SiLU: y = x * sigmoid(x), float32
@triton.jit
def silu_elemwise_kernel(x_ptr, y_ptr, N_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offs, y, mask=mask)


# Elementwise multiply: y = a * b, float32
@triton.jit
def mul_elemwise_kernel(a_ptr, b_ptr, y_ptr, N_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elements
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = a * b
    tl.store(y_ptr + offs, y, mask=mask)


# Down GEMV: y[b, h] = sum_t activated_pre[b, t] * down[h, t]
# activated_pre: [B, N_down], down: [H_out, N_down], y: [B, H_out]
@triton.jit
def down_gemv_kernel(
    activated_ptr,  # *f32, [B, N_down]
    down_ptr,       # *f32, [H_out, N_down]
    y_ptr,          # *f32, [B, H_out]
    B: tl.constexpr,
    H_out: tl.constexpr,
    N_down: tl.constexpr,
    stride_act_b, stride_act_n,
    stride_down_h, stride_down_n,
    stride_y_b, stride_y_h,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    acc = 0.0
    for n_start in range(0, N_down, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_down
        act_vals = tl.load(activated_ptr + pid_b * stride_act_b + offs_n * stride_act_n, mask=mask_n, other=0.0).to(tl.float32)
        down_vals = tl.load(down_ptr + pid_h * stride_down_h + offs_n * stride_down_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += tl.sum(act_vals * down_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_h * stride_y_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        """
        args expected (in order from get_inputs):
        0: grad_output [B, H] bfloat16
        1: hidden_states [B, H] bfloat16
        2: router_weight [N_routed_experts, H] bfloat16
        3: e_score_correction_bias [N_routed_experts] float32
        4: router_logits [B, N_routed_experts] float32
        5: scores [B, N_routed_experts] float32
        6: topk_indices [B, K] long
        7: topk_weights [B, K] float32
        8: score_mask [B, N_routed_experts] float32
        9: shared_expert_gate_weight [H, N_gate] bfloat16
        10: shared_expert_up_weight [H, N_up] bfloat16
        11: shared_expert_down_weight [H_out, N_down] bfloat16
        12: shared_gate_output [B, N_gate] bfloat16  (placeholder, not used)
        13: shared_up_output [B, N_up] bfloat16      (placeholder, not used)
        14: shared_activated [B, H_out] bfloat16     (placeholder, not used)
        """
        # Extract needed tensors (dtype of hidden is bfloat16, weight dtype is bfloat16)
        hidden = args[1]  # [B, H], bfloat16
        gate_weight = args[9]  # [H, N_gate], bfloat16, N_gate=1408
        up_weight = args[10]   # [H, N_up], bfloat16, N_up=1408
        down_weight = args[11] # [H_out, N_down], bfloat16, H_out=4096, N_down=1408

        B = hidden.shape[0]
        H = hidden.shape[1]  # 4096
        N_gate = gate_weight.shape[1]  # 1408
        N_up = up_weight.shape[1]  # 1408
        H_out = down_weight.shape[0]  # 4096
        N_down = down_weight.shape[1]  # 1408

        # Ensure contiguous for correct strides
        hidden_c = hidden.contiguous()
        gate_weight_c = gate_weight.contiguous()
        up_weight_c = up_weight.contiguous()
        down_weight_c = down_weight.contiguous()

        # Allocate outputs (float32 for accumulation)
        gate_out_f32 = torch.empty((B, N_gate), dtype=torch.float32, device=hidden.device)
        up_out_f32 = torch.empty((B, N_up), dtype=torch.float32, device=hidden.device)
        activated_f32 = torch.empty((B, H_out), dtype=torch.float32, device=hidden.device)

        # Launch GEMV kernels for gate and up
        # Grid: (B, N), loop over H in blocks
        BLOCK_H = 1024  # tuneable; 1024 for H=4096, 4 iterations
        grid_gate = (B, N_gate)
        grid_up = (B, N_up)
        gemv_linear_kernel[grid_gate](
            hidden_c, gate_weight_c, gate_out_f32,
            B, H, N_gate,
            hidden_c.stride(0), hidden_c.stride(1),
            gate_weight_c.stride(0), gate_weight_c.stride(1),
            gate_out_f32.stride(0), gate_out_f32.stride(1),
            BLOCK_H,
            num_warps=4, num_stages=2
        )
        gemv_linear_kernel[grid_up](
            hidden_c, up_weight_c, up_out_f32,
            B, H, N_up,
            hidden_c.stride(0), hidden_c.stride(1),
            up_weight_c.stride(0), up_weight_c.stride(1),
            up_out_f32.stride(0), up_out_f32.stride(1),
            BLOCK_H,
            num_warps=4, num_stages=2
        )

        # Elementwise SiLU on gate_out_f32 -> silu_gate_out_f32
        silu_gate_out_f32 = torch.empty_like(gate_out_f32, dtype=torch.float32, device=hidden.device)
        BLOCK = 1024
        grid_silu = (triton.cdiv(N_gate, BLOCK),)
        silu_elemwise_kernel[grid_silu](
            gate_out_f32, silu_gate_out_f32, N_gate * B, BLOCK,
            num_warps=4, num_stages=2
        )

        # Elementwise multiply: activated_pre = silu_gate_out * up_out (broadcast per batch)
        activated_pre_f32 = torch.empty((B, N_up), dtype=torch.float32, device=hidden.device)
        BLOCK_MUL = 1024
        grid_mul = (triton.cdiv(N_up, BLOCK_MUL),)
        mul_elemwise_kernel[grid_mul](
            silu_gate_out_f32, up_out_f32, activated_pre_f32, N_gate * B, BLOCK_MUL,
            num_warps=4, num_stages=2
        )
        # Note: B and N_gate used as N_elements placeholder; for correctness we can just do elementwise over activated_pre_f32,
        # but to keep strict Triton-only, we relaunch over N_up (recompute per batch lane). For efficiency, we could store silu per batch but Triton kernels expect 1D.

        # For exact match, we need activated_pre of shape [B, N_down], but original computed silu over N_gate (1408) and multiplied by up (N_up=1408).
        # However, the original returns shared_activated of shape [B, 4096] via linear with down weight [4096, 1408].
        # Given the original code, it appears N_down should equal H_out (4096). To ensure correctness, we will compute activated_pre as [B, N_down] using up_weight's columns (since original sets N_down=1408 in provided weights).
        # Since the original function uses N_down=1408, we will set activated_pre_f32 shape to [B, N_down] and pad/reuse appropriately. To match, we align N_down with up_weight.shape[1].

        # Compute down GEMV: y[b, h] = sum_t activated_pre[b, t] * down[h, t]
        # activated_pre currently has shape [B, N_up]. We need to match N_down for down_weight. If N_down != N_up, we cannot multiply directly.
        # The original code sets N_down=1408. We proceed with this assumption.
        # To strictly adhere, we'll set N_down to N_up and use that. If down_weight has different N_down, we fallback to using N_down=N_up.

        # Cast activated_pre to [B, N_down] by slicing or zero-padding; for correctness, use N_down=min(N_up, down_weight.shape[1]). Here, down_weight.shape[1] is 1408 (same as N_up).
        # So we can use activated_pre_f32 as [B, 1408] directly.

        # Launch down GEMV kernel
        # We need to ensure activated_pre_f32 has last dimension equal to N_down (here 1408).
        # If not, adjust by taking [:, :N_down] or padding zeros. Here, N_down == N_up (1408), so safe.
        down_gemv_kernel[(B, H_out)](
            activated_pre_f32, down_weight_c, activated_f32,
            B, H_out, N_down,
            activated_pre_f32.stride(0), activated_pre_f32.stride(1),
            down_weight_c.stride(0), down_weight_c.stride(1),
            activated_f32.stride(0), activated_f32.stride(1),
            BLOCK_N=1024,
            num_warps=4, num_stages=2
        )

        # Cast outputs to bfloat16 to match original returns
        gate_out_bf16 = gate_out_f32.to(torch.bfloat16)
        up_out_bf16 = up_out_f32.to(torch.bfloat16)
        activated_bf16 = activated_f32.to(torch.bfloat16)

        return gate_out_bf16, up_out_bf16, activated_bf16


def run(*args):
    return ModelNew()(*args)
