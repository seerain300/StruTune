import torch
import triton
import triton.language as tl


@triton.jit
def _rfft_real_imag_kernel_2d(
    x_ptr,              # *f32, input x: (B, C, S) contiguous along last dim, we pass as (B*C, S) view
    out_real_ptr,       # *f32, output real part: (B, C, S+1)
    out_imag_ptr,       # *f32, output imag part: (B, C, S+1)
    B: tl.int32,        # batch size
    C: tl.int32,        # channels
    S: tl.int32,        # seqlen
    BLOCK_K: tl.constexpr,
):
    # 2D launch: pid0 over B*C, pid1 over j in [0, S] (i.e., output upper indices)
    pid0 = tl.program_id(0)  # index over batch*channels
    j = tl.program_id(1)     # output index 0..S

    # Map pid0 to (b, c)
    b = pid0 // C
    c = pid0 % C

    # Base offsets
    base_x = (b * C + c) * S
    base_out = (b * C + c) * (S + 1)

    # Accumulators
    real_acc = 0.0
    imag_acc = 0.0

    # Normalize factor
    inv_n = 1.0 / (2.0 * S)

    # Iterate over k in chunks: k in [0, 2*S-1]
    for k0 in range(0, 2 * S, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < (2 * S)

        # Load xp[k]: for k < S, xp[k] = x[k]; else xp[k] = 0
        # x_ptr is a flat pointer; we access x[b, c, k] via base_x + k.
        x_vals = tl.load(x_ptr + base_x + k, mask=mask_k & (k < S), other=0.0)

        # Compute angle = 2*pi*j*k/(2*S)
        angle = 2.0 * 3.141592653589793 * (j * k) / (2.0 * S)

        cos_term = tl.cos(angle)
        sin_term = tl.sin(angle)

        # Accumulate sums (reduce vector to scalar)
        real_acc += tl.sum(x_vals * cos_term, axis=0)
        imag_acc += tl.sum(x_vals * sin_term, axis=0)

    # Normalize
    real_val = real_acc * inv_n
    imag_val = imag_acc * inv_n

    # Store results
    out_real_idx = base_out + j
    out_imag_idx = base_out + j
    tl.store(out_real_ptr + out_real_idx, real_val)
    # imag at j==0 should be 0; store 0.0 explicitly for safety
    if j == 0:
        tl.store(out_imag_ptr + out_imag_idx, 0.0)
    else:
        tl.store(out_imag_ptr + out_imag_idx, imag_val)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (B, C, S)
        assert x.dim() == 3, "Input must be a 3D tensor (batch, channels, seqlen)"
        B, C, S = x.shape

        # Cast to float32 (original did x.to(torch.float32))
        x_f32 = x.to(torch.float32)

        # Make contiguous; Triton expects flat memory access
        x_f32 = x_f32.contiguous()

        # Allocate outputs: (B, C, S+1)
        out_real = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernel with 2D grid: (B*C, S+1)
        grid = (B * C, S + 1)

        # Choose BLOCK_K; 128 is a reasonable default
        BLOCK_K = 128

        _rfft_real_imag_kernel_2d[grid](
            x_f32, out_real, out_imag,
            B, C, S,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=2,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
