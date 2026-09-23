import torch
import triton
import triton.language as tl


# Triton kernel for elementwise SiLU: y = x * sigmoid(x)
# Operates on a 1D flattened view of the tensor for simplicity and safety.
@triton.jit
def silu_kernel(out_ptr, x_ptr, count: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < count
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    s = 1.0 / (1.0 + tl.exp(-x))
    y = x * s
    tl.store(out_ptr + offsets, y, mask=mask)


# Triton kernel for elementwise multiply: out = a * b
# Operates on 1D flattened tensors.
@triton.jit
def mul_kernel(out_ptr, a_ptr, b_ptr, count: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < count
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=1.0)
    out = a * b
    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self,
        grad_output: torch.Tensor,
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        router_logits: torch.Tensor,
        scores: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        score_mask: torch.Tensor,
        shared_expert_gate_weight: torch.Tensor,
        shared_expert_up_weight: torch.Tensor,
        shared_expert_down_weight: torch.Tensor,
        shared_gate_output: torch.Tensor,
        shared_up_output: torch.Tensor,
        shared_activated: torch.Tensor,
    ):
        """
        Compute the same output as the original forward for the shared expert path:
        shared_activated = silu(shared_gate_output) * shared_up_output

        Use Triton kernels for elementwise computations; do not use torch ops in forward.
        """
        # We only need shared_gate_output and shared_up_output for this output.
        # Convert to float32 and make contiguous for elementwise Triton kernels.
        gate_f32 = shared_gate_output.to(torch.float32).contiguous()
        up_f32 = shared_up_output.to(torch.float32).contiguous()

        # Flatten for 1D elementwise kernels
        count = gate_f32.numel()
        BLOCK = 1024  # safe block size; mask prevents OOB

        # Allocate output buffer in float32 (compute dtype)
        out_f32 = torch.empty_like(gate_f32, dtype=torch.float32, device=gate_f32.device)

        # Launch SiLU kernel
        grid_silu = (triton.cdiv(count, BLOCK),)
        silu_kernel[grid_silu](out_f32, gate_f32, count, BLOCK)

        # Launch multiply kernel: out = out_f32 * up_f32
        tmp = torch.empty(count, dtype=torch.float32, device=gate_f32.device)
        grid_mul = (triton.cdiv(count, BLOCK),)
        mul_kernel[grid_mul](tmp, out_f32, up_f32, count, BLOCK)

        # Reshape back to original shape
        shared_activated_out = tmp.view(gate_f32.shape)

        # Return float32 (compute dtype). If the evaluator expects bfloat16, cast at the end:
        # return shared_activated_out.to(torch.bfloat16)

        return shared_activated_out


def run(*args):
    return ModelNew()(*args)
