import torch
import triton
import triton.language as tl


@triton.jit
def apply_threshold_relu_to_bf16(in_ptr, out_ptr, mean_ptr, std_ptr, z_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per (b, s) row
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Load per-row mean, std (fp32), and scalar z (fp32)
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    z = tl.load(z_ptr)  # 1-element tensor, scalar fp32

    # Compute threshold
    thr = mean + std * z  # fp32

    # Base offset for this row in flattened [B, S, H] layout
    base = b * S * H + s * H

    offsets = tl.arange(0, BLOCK)
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        # Load input (bf16), compute y = max(x - thr, 0.0) in fp32
        x_bf = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        x = x_bf.to(tl.float32)
        y = x - thr
        y = tl.maximum(y, 0.0)
        # Store as bf16
        tl.store(out_ptr + base + idx, y.to(tl.bfloat16), mask=mask)
        i += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float):
        """
        Triton-only implementation:
        - Compute mean and std along feature dimension (unbiased=False) using PyTorch to match behavior.
        - Compute inverse normal CDF for target_sparsity using torch.special.erfinv.
        - Apply thresholding with ReLU via Triton kernel.
        """
        # If no sparsity requested, return original
        if target_sparsity == 0.0:
            return inputs

        # Ensure input is bfloat16 (original code returns bfloat16)
        x = inputs.to(torch.bfloat16)
        B, S, H = x.shape
        device = x.device

        # Compute mean and std along last dim with keepdim=True -> shape [B, S, 1]
        # Cast to fp32 for accurate statistics
        x_f32 = x.to(torch.float32)
        mean = x_f32.mean(dim=-1, keepdim=True)
        std = x_f32.std(dim=-1, unbiased=False, keepdim=True)  # population std

        # Compute inverse normal CDF: z = sqrt(2) * erfinv(2*p - 1)
        p = torch.tensor(target_sparsity, dtype=torch.float32, device=device)
        z = torch.sqrt(torch.tensor(2.0, device=device)) * torch.special.erfinv((2.0 * p) - 1.0)
        z_scalar = z.reshape(1)  # 1-element tensor to pass to Triton

        # Prepare output
        out = torch.empty_like(x)

        # Launch Triton kernel: one program per (b, s) row
        grid = (B * S,)
        apply_threshold_relu_to_bf16[grid](x, out, mean, std, z_scalar, B, S, H, BLOCK=2048)

        return out


def run(*args):
    return ModelNew()(*args)
