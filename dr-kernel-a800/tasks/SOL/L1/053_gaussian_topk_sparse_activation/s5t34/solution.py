import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def reduce_row_sum_sumsq_kernel(
        x_ptr,            # *const float32, flattened input
        sums_ptr,         # *float32, length B*S
        sums2_ptr,        # *float32, length B*S
        F,                # feature size
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per row (row = (batch, seq))
        pid = tl.program_id(axis=0)
        row_start = pid * F
        local_sum = 0.0
        local_sumsq = 0.0

        for offs in range(0, F, BLOCK_SIZE):
            idx = offs + tl.arange(0, BLOCK_SIZE)
            mask = idx < F
            ptrs = x_ptr + row_start + idx
            vals = tl.load(ptrs, mask=mask, other=0.0)
            local_sum += tl.sum(vals, axis=0)
            local_sumsq += tl.sum(vals * vals, axis=0)

        tl.store(sums_ptr + pid, local_sum)
        tl.store(sums2_ptr + pid, local_sumsq)

    @triton.jit
    def compute_mean_std_kernel(
        sums_ptr,         # *float32, length B*S
        sums2_ptr,        # *float32, length B*S
        mean_ptr,         # *float32, length B*S
        std_ptr,          # *float32, length B*S
        F,                # feature size
    ):
        pid = tl.program_id(axis=0)
        sum = tl.load(sums_ptr + pid)
        sum2 = tl.load(sums2_ptr + pid)
        mean = sum / F
        var = sum2 / F - mean * mean
        var = tl.maximum(var, 0.0)
        std = tl.sqrt(var)
        tl.store(mean_ptr + pid, mean)
        tl.store(std_ptr + pid, std)

    @triton.jit
    def inv_erf_kernel(
        out_ptr,          # *float32, scalar output
        x_ptr,            # *const float32, scalar input (target_sparsity)
        A1, A2, A3, A4, A5,
        B1, B2, B3, B4, B5,
        C1, C2, C3, C4, C5, C6,
        D1, D2, D3, D4,
    ):
        # Load scalar target_sparsity
        sp = tl.load(x_ptr)
        # Abramowitz and Stegun 7.1.26 approximation for inverse erf
        # For x in [0, 1]
        # t = 1 - sp
        t = 1.0 - sp
        # Piecewise approximation
        p_low = 0.02425
        p_high = 1.0 - p_low

        # Lower region
        q = tl.sqrt(-2.0 * tl.log(t))
        res_low = (((((C1 * q + C2) * q + C3) * q + C4) * q + C5) * q + C6) / \
                  ((((D1 * q + D2) * q + D3) * q + D4) * q + 1.0)

        # Central region
        q_mid = sp - 0.5
        r_mid = q_mid * q_mid
        poly = (((((A1 * r_mid + A2) * r_mid + A3) * r_mid + A4) * r_mid + A5) * r_mid + 6.0) * q_mid
        poly2 = (((((B1 * r_mid + B2) * r_mid + B3) * r_mid + B4) * r_mid + B5) * r_mid + 1.0)
        res_mid = poly / poly2

        # Upper region
        q_up = tl.sqrt(-2.0 * tl.log(1.0 - sp))
        res_up = -(((((C1 * q_up + C2) * q_up + C3) * q_up + C4) * q_up + C5) * q_up + C6) / \
                 ((((D1 * q_up + D2) * q_up + D3) * q_up + D4) * q_up + 1.0)

        # Combine
        cond_low = sp < p_low
        cond_mid = (sp >= p_low) & (sp <= p_high)
        cond_up = sp > p_high
        res = tl.where(cond_low, res_low, 0.0)
        res = tl.where(cond_mid, res_mid, res)
        res = tl.where(cond_up, res_up, res)

        # Store inv_erf(sp) * sqrt(2)
        tl.store(out_ptr, res * tl.sqrt(2.0))

    @triton.jit
    def sparsify_relu_kernel(
        x_ptr,            # *const float32 input
        out_ptr,          # *float32 output
        mean_ptr,         # *const float32, shape [B*S]
        std_ptr,          # *const float32, shape [B*S]
        threshold_scale,  # scalar float32 multiplier for threshold (inv_erf(target_sparsity) * sqrt(2))
        total_elems,      # int32
        BLOCK_SIZE: tl.constexpr,
    ):
        # 1D grid over total elements
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < total_elems

        # Compute (b, s) and f from linear index
        SF = 1  # placeholder to avoid division by zero
        # We cannot use shape constexpr in Triton easily; instead, we rely on the grid size to match total_elems.
        # For each offs, compute b, s, f via division/mod by S and F. We pass S and F to the kernel.
        # However, Triton requires S and F as constexpr for indexing; so we re-launch with proper grid computed from S and F in host.
        # Here, we assume host sets grid = (triton.cdiv(total_elems, BLOCK_SIZE),) and we recompute b,s,f as below:
        # Note: Triton does not allow dynamic shape retrieval; we must pass S and F as constexpr. To keep code simple, we assume host passes S and F to the kernel via args. But Triton does not support this.
        # Therefore, we instead compute using linear indexing only: we flatten [B,S,F] into 1D. We need S and F to compute indices.
        # Since S and F are not directly accessible, we instead rely on host to launch this kernel only when S and F are known, and we compute indices in the kernel using assumed layout.
        # To keep correctness, we restructure: use two kernels: reduction and mean/std; then a third kernel that applies sparsity using mean/std arrays.
        # In this file, we keep host-side logic minimal; we do not use torch for math. The host will pass S and F via function arguments, and we assume total_elems == B*S*F.

        # Reconstruct b, s, f via division/mod. We need S and F as runtime ints. Triton doesn't allow retrieving them; so we pass them via kernel launch, which requires constexpr. Hence, we need to avoid this complexity.

        # Simplify: host will precompute mean and std and pass pointers. We will not compute indices here; instead, we'll implement a simpler sparsity kernel that assumes contiguous [B,S,F] layout and applies threshold using loaded mean/std for each element based on its linear index. But we need per-(b,s) threshold. Triton does not support indirect indexing with runtime arrays efficiently here. Therefore, we implement sparsity with per-element threshold as zero, which is incorrect. We need to fix this by computing b,s,f indices correctly.

        # Fix: We restructure kernels. We will not write sparsify kernel here. We instead provide a complete implementation below that correctly computes b,s,f indices using constexpr S and F passed to the kernel via launch. For simplicity, we redefine the sparsify kernel properly now.

        # Correct sparsify kernel (requires S and F as constexpr):
        # Here we redefine the sparsify kernel with S and F as constexpr.

        # Note: We cannot redefine here since Triton JIT requires full kernel definition. Therefore, we include the correct sparsify kernel definition below. For now, we provide the forward logic that launches this kernel correctly.

        # The following block is a placeholder. Actual sparsify kernel is defined below this comment.

        pass  # Placeholder to satisfy Triton JIT; actual kernel is defined below.


# The correct sparsify kernel definition:
@triton.jit
def sparsify_relu_kernel_proper(
    x_ptr,            # *const float32 input flattened
    out_ptr,          # *float32 output flattened
    mean_ptr,         # *const float32, shape [B*S]
    std_ptr,          # *const float32, shape [B*S]
    threshold_scale,  # scalar float32 multiplier for threshold (inv_erf(target_sparsity) * sqrt(2))
    B: tl.constexpr,  # batch size (constexpr)
    S: tl.constexpr,  # seq_len (constexpr)
    F: tl.constexpr,  # feature size (constexpr)
    BLOCK_SIZE: tl.constexpr,
):
    # 1D grid over total elements
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = B * S * F
    mask = offs < total

    # Compute (b, s, f) from linear index
    SF = S * F
    b = offs // SF
    rem = offs % SF
    s = rem // F
    f = rem % F

    x_ptrs = x_ptr + offs
    out_ptrs = out_ptr + offs

    # Load input
    x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

    # Load mean and std for each (b, s) and compute threshold
    row_id = b * S + s
    mean = tl.load(mean_ptr + row_id, mask=mask, other=0.0)
    std = tl.load(std_ptr + row_id, mask=mask, other=0.0)
    threshold = mean + std * threshold_scale

    # Apply y = max(0, x - threshold)
    y = x_vals - threshold
    y = tl.maximum(y, 0.0)

    # Store
    tl.store(out_ptrs, y, mask=mask)


# -----------------------------
# ModelNew: Triton-only forward
# -----------------------------
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        assert TRITON_AVAILABLE, "Triton is not available"
        assert inputs.is_cuda, "Input must be on CUDA for Triton execution"

        # Ensure float32 and contiguous
        x = inputs.to(torch.float32).contiguous()
        B, S, F = x.shape
        device = x.device

        total_elems = B * S * F

        # 1) Reduce per-(batch, seq) to sums and sums of squares
        sums = torch.empty(B * S, dtype=torch.float32, device=device)
        sums2 = torch.empty(B * S, dtype=torch.float32, device=device)

        BLOCK_SIZE_RED = 1024
        grid_red = (B * S,)
        reduce_row_sum_sumsq_kernel[grid_red](
            x, sums, sums2, F, BLOCK_SIZE=BLOCK_SIZE_RED
        )

        # 2) Compute mean and std per (batch, seq) in Triton
        mean = torch.empty(B * S, dtype=torch.float32, device=device)
        std = torch.empty(B * S, dtype=torch.float32, device=device)

        compute_mean_std_kernel[grid_red](sums, sums2, mean, std, F)

        # 3) Compute inv_erf(target_sparsity) * sqrt(2) in Triton (Abramowitz & Stegun 7.1.26)
        inv_erf_buf = torch.empty(1, dtype=torch.float32, device=device)
        A1 = -1.109820872433686
        A2 = 1.147078267979718
        A3 = -0.822152270978925
        A4 = 0.337475330570173
        A5 = -0.084679213565134
        B1 = -4.433816845333261
        B2 = 1.935916034185939
        B3 = -0.603118321792633
        B4 = 0.287162667281933
        B5 = -0.138938950259879
        C1 = -7.784894002430293e-03
        C2 = -3.223964580411365e-01
        C3 = -2.400758277161838e+00
        C4 = -2.549732539343734e+00
        C5 = 4.374664141464968e+00
        C6 = 2.938163982698783e+00
        D1 = 7.784695709041462e-03
        D2 = 3.224671290700398e-01
        D3 = 2.445134137142996e+00
        D4 = 3.754408661907416e+00

        sp_val = torch.tensor(target_sparsity, dtype=torch.float32, device=device)
        inv_erf_kernel[(1,)](
            inv_erf_buf,
            sp_val,
            A1, A2, A3, A4, A5,
            B1, B2, B3, B4, B5,
            C1, C2, C3, C4, C5, C6,
            D1, D2, D3, D4,
        )
        threshold_scale = inv_erf_buf[0] * 1.4142135623730951  # sqrt(2)

        # 4) Apply sparsity: output = max(0, x - (mean + std * threshold_scale)) in Triton
        out_fp32 = torch.empty(total_elems, dtype=torch.float32, device=device)

        BLOCK_SIZE_POINT = 1024
        grid_point = (triton.cdiv(total_elems, BLOCK_SIZE_POINT),)
        sparsify_relu_kernel_proper[grid_point](
            x, out_fp32, mean, std, threshold_scale,
            B, S, F, BLOCK_SIZE=BLOCK_SIZE_POINT
        )

        # Reshape and return as bfloat16 to match original behavior
        out = out_fp32.view(B, S, F).to(torch.bfloat16)
        return out


def run(*args):
    return ModelNew()(*args)
