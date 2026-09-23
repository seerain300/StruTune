import torch
import triton
import triton.language as tl


# Triton GEMV for output size N=1408: y[b, t] = sum_h hidden[b, h] * W[t, h]
# hidden: [B, H] (input dtype), W: [N=1408, H] (input dtype), y: [B, 1408] (float32)
@triton.jit
def gemv_linear_1408_kernel(
    hidden_ptr,   # *T, [B, H]
    W_ptr,        # *T, [1408, H]
    y_ptr,        # *f32, [B, 1408]
    B: tl.constexpr,
    H: tl.constexpr,        # hidden_size, e.g., 4096
    N: tl.constexpr,        # output size, 1408
    stride_h_b, stride_h_h,
    stride_W_t, stride_W_h,
    stride_y_b, stride_y_t,
    BLOCK_H: tl.constexpr,  # e.g., 256
):
    pid_b = tl.program_id(0)  # batch row
    pid_t = tl.program_id(1)  # output index t in [0, N)
    acc = 0.0
    # Loop over hidden dimension in chunks of BLOCK_H
    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        # Load segment of hidden for this batch row
        h_vals = tl.load(hidden_ptr + pid_b * stride_h_b + offs_h * stride_h_h, mask=mask_h, other=0.0)
        # Load segment of W (weight vector) for output index pid_t
        w_vals = tl.load(W_ptr + pid_t * stride_W_t + offs_h * stride_W_h, mask=mask_h, other=0.0)
        # Accumulate dot product in float32
        acc += tl.sum(h_vals.to(tl.float32) * w_vals.to(tl.float32), axis=0)
    # Store result
    tl.store(y_ptr + pid_b * stride_y_b + pid_t * stride_y_t, acc)


# Triton elementwise kernel: y = silu(a) * b, where a, b: [B, N] (f32), y: [B, N] (f32)
@triton.jit
def silu_mul_elemwise_kernel(a_ptr, b_ptr, y_ptr, total_elems: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-a))
    y = a * sig * b
    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # This forward must NOT use torch ops on device tensors; all math via Triton kernels.
        # The original Model.forward returns (shared_gate_output, shared_up_output, shared_activated).
        # We will compute:
        # 1) gate_output = F.linear(hidden, shared_expert_gate_weight) -> [B, 1408]
        # 2) up_output    = F.linear(hidden, shared_expert_up_weight)    -> [B, 1408]
        # 3) activated_pre = F.silu(gate_output) * up_output             -> [B, 1408]
        # and return them, casting to bfloat16 at the end.

        # Extract inputs:
        # We assume args are passed in the same order as the original function:
        # grad_output, hidden_states, router_weight, e_score_correction_bias, 
        # router_logits, scores, topk_indices, topk_weights, score_mask,
        # shared_expert_gate_weight, shared_expert_up_weight, shared_expert_down_weight,
        # shared_gate_output, shared_up_output, shared_activated.
        # We only need hidden, gate_weight, up_weight. The other args are ignored as the original returns only 3 outputs.

        hidden = args[0]
        gate_weight = args[8]  # shared_expert_gate_weight, shape [H=4096, N_gate=1408], dtype likely bfloat16
        up_weight = args[9]    # shared_expert_up_weight, shape [H=4096, N_up=1408], dtype likely bfloat16

        # Ensure device and contiguity
        device = hidden.device
        hidden_c = hidden.contiguous()
        gate_weight_c = gate_weight.contiguous()
        up_weight_c = up_weight.contiguous()

        B = hidden_c.shape[0]
        H = hidden_c.shape[1]  # 4096
        N = 1408  # output size

        # 1) Compute gate_output [B, 1408] (f32) via Triton GEMV
        gate_out_f32 = torch.empty((B, N), dtype=torch.float32, device=device)
        grid_gate = (B, N)
        gemv_linear_1408_kernel[grid_gate](
            hidden_c, gate_weight_c, gate_out_f32,
            B, H, N,
            hidden_c.stride(0), hidden_c.stride(1),
            gate_weight_c.stride(0), gate_weight_c.stride(1),
            gate_out_f32.stride(0), gate_out_f32.stride(1),
            BLOCK_H=256,
            num_warps=4,
        )

        # 2) Compute up_output [B, 1408] (f32) via Triton GEMV
        up_out_f32 = torch.empty((B, N), dtype=torch.float32, device=device)
        grid_up = (B, N)
        gemv_linear_1408_kernel[grid_up](
            hidden_c, up_weight_c, up_out_f32,
            B, H, N,
            hidden_c.stride(0), hidden_c.stride(1),
            up_weight_c.stride(0), up_weight_c.stride(1),
            up_out_f32.stride(0), up_out_f32.stride(1),
            BLOCK_H=256,
            num_warps=4,
        )

        # 3) Compute activated_pre = silu(gate_out) * up_out elementwise (f32)
        total_elems = B * N
        activated_f32 = torch.empty(total_elems, dtype=torch.float32, device=device)
        silu_mul_elemwise_kernel[(triton.cdiv(total_elems, 4096),)](
            gate_out_f32, up_out_f32, activated_f32,
            total_elems,
            4096,  # BLOCK for flattened vector
            num_warps=4,
        )
        # Reshape to [B, N]
        activated_f32 = activated_f32.view(B, N)

        # Return exactly three outputs, cast to bfloat16 to match original dtype expectations.
        shared_gate_output = gate_out_f32.to(torch.bfloat16)
        shared_up_output = up_out_f32.to(torch.bfloat16)
        shared_activated = activated_f32.to(torch.bfloat16)

        return shared_gate_output, shared_up_output, shared_activated


def run(*args):
    return ModelNew()(*args)
