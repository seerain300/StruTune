import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# -----------------------------
# Triton kernels
# -----------------------------
if TRITON_AVAILABLE:
    @triton.jit
    def reduce_sum_sumsq_kernel(
        x_ptr,                  # *const float32, flattened input
        sums_ptr,               # *float32, length B*S
        sums2_ptr,              # *float32, length B*S
        B, S, F,                # int32, dims
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        b = pid // S
        s = pid % S
        row_start = (b * S + s) * F

        local_sum = 0.0
        local_sumsq = 0.0

        for offs in range(0, F, BLOCK_SIZE):
            idx = offs + tl.arange(0, BLOCK_SIZE)
            mask = idx < F
            ptrs = x_ptr + row_start + idx
            vals = tl.load(ptrs, mask=mask, other=0.0)
            local_sum += tl.sum(vals, axis=0)
            local_sumsq += tl.sum(vals * vals, axis=0)

        tl.atomic_add(sums_ptr + pid, local_sum)
        tl.atomic_add(sums2_ptr + pid, local_sumsq)

    @triton.jit
    def compute_mean_std_kernel(
        sums_ptr,               # *float32, length B*S
        sums2_ptr,              # *float32, length B*S
        mean_ptr,               # *float32, length B*S
        std_ptr,                # *float32, length B*S
        F: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        total = tl.load(sums_ptr + pid)
        total2 = tl.load(sums2_ptr + pid)
        mean = total / F
        var = total2 / F - mean * mean
        var = tl.maximum(var, 0.0)  # guard against tiny negative due to numerical error
        std = tl.sqrt(var)
        tl.store(mean_ptr + pid, mean)
        tl.store(std_ptr + pid, std)

    @triton.jit
    def sparsify_relu_kernel(
        x_ptr,                  # *const float32 (input)
        out_ptr,                # *bfloat16 (output)
        threshold_ptr,          # *const float32, shape [B*S]
        total_elems: tl.constexpr,
        B, S, F,                # int32
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        start = pid * BLOCK_SIZE
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < total_elems

        # Compute (b, s) for each element
        B_dim = S * F
        s = idx // B_dim
        b = (idx // F) % B
        bs = b * S + s

        # Gather input
        in_ptrs = x_ptr + idx
        x_vals = tl.load(in_ptrs, mask=mask, other=0.0)

        # Gather threshold for each (b, s)
        th_ptrs = threshold_ptr + bs
        th_vals = tl.load(th_ptrs, mask=mask, other=0.0)  # per-element threshold

        diff = x_vals - th_vals
        out_vals = tl.maximum(diff, 0.0)

        # Store bfloat16 output
        out_ptrs = out_ptr + idx
        tl.store(out_ptrs, out_vals, mask=mask)


# -----------------------------
# Entry point module
# -----------------------------
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # Expect single tensor input: [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise ValueError("ModelNew.forward expects a single tensor input of shape [batch_size, seq_len, intermediate_size].")
        inputs = args[0]

        if not TRITON_AVAILABLE or not inputs.is_cuda:
            # Fallback to pure PyTorch if Triton/CUDA not available
            x = inputs.to(torch.float32)
            B, S, F = x.shape
            mean = torch.mean(x, dim=-1, keepdim=True)
            std = torch.std(x, dim=-1, keepdim=True, unbiased=False)
            # Default target_sparsity; keep it configurable externally if needed
            sp = 0.5
            # Normal inverse CDF: ndtri(p) = sqrt(2) * erfinv(2p - 1)
            try:
                from torch.special import erfinv
                z = torch.sqrt(torch.tensor(2.0, device=x.device)) * erfinv((2.0 * sp - 1.0).to(x.device))
                multiplier = float(z.item())
            except Exception:
                multiplier = 0.0
            cutoff = mean + std * multiplier
            sparse = torch.relu(x - cutoff)
            return sparse.to(torch.bfloat16)

        # Triton path
        x = inputs.to(torch.float32).contiguous()  # compute stats in float32
        B, S, F = x.shape
        total_rows = B * S
        total_elems = B * S * F

        # 1) Reduce per-row sums and sums of squares
        sums = torch.zeros(total_rows, dtype=torch.float32, device=x.device)
        sums2 = torch.zeros(total_rows, dtype=torch.float32, device=x.device)

        BLOCK_SIZE = 1024
        reduce_sum_sumsq_kernel[(total_rows,)](
            x, sums, sums2, B, S, F, BLOCK_SIZE=BLOCK_SIZE
        )

        # 2) Compute mean and std per (b, s) in Triton
        mean = torch.empty(total_rows, dtype=torch.float32, device=x.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=x.device)
        compute_mean_std_kernel[(total_rows,)](
            sums, sums2, mean, std, F=F
        )
        mean = mean.view(B, S)  # [B, S]
        std = std.view(B, S)    # [B, S]

        # 3) Build threshold tensor [B, S, 1] and compute multiplier (scalar)
        sp = 0.5  # default target_sparsity; adjust externally if needed
        try:
            from torch.special import erfinv
            z = torch.sqrt(torch.tensor(2.0, device=x.device)) * erfinv((2.0 * sp - 1.0).to(x.device))
            multiplier = float(z.item())
        except Exception:
            multiplier = 0.0
        cutoff = mean + std * multiplier  # [B, S]
        threshold = cutoff.unsqueeze(-1).contiguous()  # [B, S, 1]

        # 4) Apply sparsification: output = max(0, x - threshold) in Triton
        out = torch.empty((B, S, F), dtype=torch.bfloat16, device=x.device)

        BLOCK_SIZE_POINT = 1024
        grid = (triton.cdiv(total_elems, BLOCK_SIZE_POINT),)
        sparsify_relu_kernel[grid](
            x, out, threshold, total_elems, B, S, F, BLOCK_SIZE=BLOCK_SIZE_POINT
        )

        return out


def run(*args):
    return ModelNew()(*args)
