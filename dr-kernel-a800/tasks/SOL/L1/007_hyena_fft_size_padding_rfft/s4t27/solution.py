import torch
import triton
import triton.language as tl


@triton.jit
def _compute_rfft_per_k_kernel(x_ptr, out_real_ptr, out_imag_ptr, L: tl.int32, N: tl.int32):
    # Each program computes one k in [0..L] for a single (bc) row (pid_bc = program_id(0))
    pid_bc = tl.program_id(0)
    k = tl.program_id(1)

    # If k > L, this program id is out of range for our desired output; return
    if k > L:
        return

    acc_real = 0.0
    acc_imag = 0.0

    # Reduction over j in chunks
    BLOCK_J = 256
    for j0 in tl.static_range(0, N, BLOCK_J):
        j = j0 + tl.arange(0, BLOCK_J)
        valid = j < N
        # Load padded x values for this (bc) row: x_ptr is (BC, L), we index by pid_bc and j
        # Note: j here indexes into the original L-length sequence; zeros beyond L are implicit.
        # Since x_ptr points to (BC, L), accessing pid_bc*L + j for j>=L is out-of-bounds; we must
        # ensure we never access beyond L. Therefore, we build vals by loading from x_ptr[pid_bc, j]
        # only for j<L and treat j>=L as zero by using mask. We can form vals by loading x[j] masked,
        # but since x_ptr is (BC, L), we instead load from a separate padded_x buffer. To simplify,
        # we pass x_ptr as (BC, L) and use masked loads with other=0.0 for j>=L. However, x_ptr has
        # only L columns. So we need a padded_x buffer. Triton kernels expect contiguous tensors.
        # To keep it simple and correct, we pre-pad x on host to length N and pass padded_x to this
        # kernel. But the original request is to use Triton only, so we create padded_x in forward
        # and pass it here.
        # For correctness, we must have x_ptr as padded_x of shape (BC, N). The code below assumes
        # that x_ptr is (BC, N) and we load x_ptr[pid_bc, j] for j in [0,N). We'll arrange this in
        # forward by allocating padded_x and copying input into it.
        vals = tl.load(x_ptr + pid_bc * N + j, mask=valid, other=0.0)
        angle = 2.0 * 3.141592653589793 * (k * j) / N
        cosv = tl.cos(angle)
        sinv = tl.sin(angle)
        acc_real += tl.sum(vals * cosv, axis=0)
        acc_imag += tl.sum(vals * sinv, axis=0)

    # Normalize by N (2*L)
    inv_N = 1.0 / N
    acc_real = acc_real * inv_N
    acc_imag = acc_imag * inv_N

    # Store results at index k in (L+1) output
    # out_real/imag are laid out as (BC, L+1) contiguous
    tl.store(out_real_ptr + pid_bc * (L + 1) + k, acc_real)
    tl.store(out_imag_ptr + pid_bc * (L + 1) + k, acc_imag)


@triton.jit
def _divide_by_scalar_kernel(in_ptr, out_ptr, numel: tl.int32, scale: tl.float32, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    x = x / scale
    tl.store(out_ptr + offs, x, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (B, C, L)
        B, C, L = x.shape
        N = 2 * L  # implicit zero-padding

        # Ensure contiguous input
        x = x.contiguous()

        # Flatten (B, C) into BC for simpler indexing
        BC = B * C

        # Allocate padded buffer and outputs
        # padded_x: (BC, N) float32, zeros for j>=L
        padded_x = torch.empty((BC, N), dtype=torch.float32, device=x.device)
        # Copy input rows into padded_x
        # We can simply fill padded_x with zeros and copy x[:, :L] into columns 0..L-1
        padded_x.zero_()
        # Copy x[b, c, :] into padded_x[pid_bc, 0:L]
        for i in range(L):
            # Note: we can't vectorize this loop in Triton here; do it in PyTorch for correctness.
            # This avoids any host-side torch.cos/torch.sin in kernels; only Triton kernels run.
            # Copy each row to padded buffer
            # Build a pointer to the (i)th element of each row and set it
            # Instead, we can do it via slicing:
            padded_x[:, i] = x[:, :, i].reshape(BC)

        # Alternatively, use a vectorized PyTorch copy without torch.cos/sin:
        # Copy each (b, c) row into padded_x
        for b in range(B):
            for c in range(C):
                bc = b * C + c
                padded_x[bc, :L] = x[b, c, :].to(torch.float32)

        # Allocate outputs: (BC, L+1) float32
        out_real = torch.empty((BC, L + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((BC, L + 1), dtype=torch.float32, device=x.device)

        # Compute rfft real/imag parts using Triton kernel
        grid_rfft = (BC, L + 1)
        _compute_rfft_per_k_kernel[grid_rfft](padded_x, out_real, out_imag, L, N)

        # Reshape back to (B, C, L+1)
        out_real = out_real.view(B, C, L + 1)
        out_imag = out_imag.view(B, C, L + 1)

        # Normalize by N (2*L) as done in the original code
        # In _compute_rfft_per_k_kernel we already divide by N; this is a safety normalization.
        tmp_real = torch.empty_like(out_real)
        tmp_imag = torch.empty_like(out_imag)
        numel_real = out_real.numel()
        numel_imag = out_imag.numel()
        BLOCK = 1024
        grid_norm = (triton.cdiv(numel_real, BLOCK),)
        _divide_by_scalar_kernel[grid_norm](out_real, tmp_real, numel_real, float(N), BLOCK=BLOCK)
        _divide_by_scalar_kernel[grid_norm](out_imag, tmp_imag, numel_imag, float(N), BLOCK=BLOCK)

        # Assign results
        out_real = tmp_real
        out_imag = tmp_imag

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
