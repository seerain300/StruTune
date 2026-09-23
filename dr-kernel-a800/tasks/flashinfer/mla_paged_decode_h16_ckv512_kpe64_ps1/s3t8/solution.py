import torch
import triton
import triton.language as tl

# Constants (compile-time) for dimensions
H = 16          # num_qo_heads
Dq = 512        # head_dim_ckv
Dp = 64         # head_dim_kpe

@triton.jit
def fused_logits_kernel(q_nope_ptr, q_pe_ptr, Kc_ptr, Kp_ptr, logits_ptr,
                         T: tl.constexpr,  # number of tokens (compile-time for tiling)
                         BLOCK_T: tl.constexpr):
    # One program per head
    h = tl.program_id(0)
    # Preload q vectors
    qn = tl.load(q_nope_ptr + h * Dq + tl.arange(0, Dq))  # [Dq]
    qp = tl.load(q_pe_ptr + h * Dp + tl.arange(0, Dp))   # [Dp]

    offs_t = tl.arange(0, BLOCK_T)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T

        # Load Kc and Kp sub-tiles
        Kc_sub = tl.load(Kc_ptr + t_idx[:, None] * Dq + tl.arange(0, Dq)[None, :], mask=mask_t[:, None], other=0.0)  # [BLOCK_T, Dq]
        Kp_sub = tl.load(Kp_ptr + t_idx[:, None] * Dp + tl.arange(0, Dp)[None, :], mask=mask_t[:, None], other=0.0)  # [BLOCK_T, Dp]

        # Compute dot-products: [BLOCK_T]
        # qn: [Dq], Kc_sub: [BLOCK_T, Dq] -> [BLOCK_T]
        dot_qn = tl.sum(qn[None, :] * Kc_sub, axis=1)
        # qp: [Dp], Kp_sub: [BLOCK_T, Dp] -> [BLOCK_T]
        dot_qp = tl.sum(qp[None, :] * Kp_sub, axis=1)

        logits_tile = dot_qn + dot_qp
        # Store logits[h, t_idx]
        tl.store(logits_ptr + h * T + t_idx, logits_tile, mask=mask_t)

@triton.jit
def softmax_row_kernel(scaled_ptr, attn_ptr, T: tl.constexpr, BLOCK_T: tl.constexpr):
    h = tl.program_id(0)
    row_ptr = scaled_ptr + h * T
    attn_out_ptr = attn_ptr + h * T

    # First pass: compute max
    m = -float('inf')
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + tl.arange(0, BLOCK_T)
        mask_t = t_idx < T
        x = tl.load(row_ptr + t_idx, mask=mask_t, other=-float('inf'))
        m = tl.maximum(m, tl.max(x, axis=0))

    # Second pass: compute sum of exp(x - m)
    sum_exp = 0.0
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + tl.arange(0, BLOCK_T)
        mask_t = t_idx < T
        x = tl.load(row_ptr + t_idx, mask=mask_t, other=-float('inf'))
        e = tl.exp(x - m)
        sum_exp += tl.sum(e, axis=0)

    inv_sum = 1.0 / sum_exp

    # Third pass: write normalized attn
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + tl.arange(0, BLOCK_T)
        mask_t = t_idx < T
        x = tl.load(row_ptr + t_idx, mask=mask_t, other=-float('inf'))
        e = tl.exp(x - m) * inv_sum
        tl.store(attn_out_ptr + t_idx, e, mask=mask_t)

@triton.jit
def lse_row_kernel(scaled_ptr, lse_ptr, T: tl.constexpr, BLOCK_T: tl.constexpr):
    h = tl.program_id(0)
    row_ptr = scaled_ptr + h * T

    m = -float('inf')
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + tl.arange(0, BLOCK_T)
        mask_t = t_idx < T
        x = tl.load(row_ptr + t_idx, mask=mask_t, other=-float('inf'))
        m = tl.maximum(m, tl.max(x, axis=0))

    sum_exp = 0.0
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + tl.arange(0, BLOCK_T)
        mask_t = t_idx < T
        x = tl.load(row_ptr + t_idx, mask=mask_t, other=-float('inf'))
        e = tl.exp(x - m)
        sum_exp += tl.sum(e, axis=0)

    lse_val = tl.log(sum_exp) + m  # logsumexp of scaled logits
    # Divide by ln(2)
    lse_val = lse_val / 0.6931471805599453  # 1 / ln(2)
    tl.store(lse_ptr + h, lse_val)

@triton.jit
def attn_matmul_kernel(attn_ptr, Kc_ptr, out_ptr,
                        T: tl.constexpr, Dq: tl.constexpr,
                        BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    h = tl.program_id(0)
    attn_row_ptr = attn_ptr + h * T
    out_row_ptr = out_ptr + h * Dq

    # Accumulator over D dimension
    acc = tl.zeros([Dq], dtype=tl.float32)

    for d_start in range(0, Dq, BLOCK_D):
        d_idx = d_start + tl.arange(0, BLOCK_D)
        mask_d = d_idx < Dq

        # For each token block, accumulate contributions into a partial acc
        # Initialize a vector acc_d of size BLOCK_D
        acc_d = tl.zeros([BLOCK_D], dtype=tl.float32)

        for t_start in range(0, T, BLOCK_T):
            t_idx = t_start + tl.arange(0, BLOCK_T)
            mask_t = t_idx < T

            attn_chunk = tl.load(attn_row_ptr + t_idx, mask=mask_t, other=0.0)  # [BLOCK_T]
            Kc_chunk = tl.load(Kc_ptr + t_idx[:, None] * Dq + d_idx[None, :], mask=mask_t[:, None] & mask_d[None, :], other=0.0)  # [BLOCK_T, BLOCK_D]

            # Multiply and reduce over tokens to update acc_d: [BLOCK_D]
            acc_d += tl.sum(attn_chunk[:, None] * Kc_chunk, axis=0)

        # Write partial to full output vector at positions [d_start:d_start+BLOCK_D]
        tl.store(out_row_ptr + d_idx, acc_d, mask=mask_d)

    # The loop updates acc piecewise; store full acc at the end
    # Since we accumulated acc_d and stored them per block, acc is already complete across Dq.
    # Ensure full acc is written: we already did via acc_d partials in the loop structure.
    # However, acc wasn't updated; instead we wrote acc_d each block. To ensure full acc is complete,
    # we can simply not store acc explicitly; the per-block stores cover the whole Dq.
    # (acc is only used as a scratch variable, but Triton requires final write; we will restructure below)

# Correct attn_matmul_kernel without using acc (avoid confusion): compute full output vector directly by per-block writes.

@triton.jit
def attn_matmul_kernel_fixed(attn_ptr, Kc_ptr, out_ptr,
                             T: tl.constexpr, Dq: tl.constexpr,
                             BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    h = tl.program_id(0)
    attn_row_ptr = attn_ptr + h * T
    out_row_ptr = out_ptr + h * Dq

    # Iterate over Dq in tiles of BLOCK_D and for each, compute the dot over T and store the partial
    for d_start in range(0, Dq, BLOCK_D):
        d_idx = d_start + tl.arange(0, BLOCK_D)
        mask_d = d_idx < Dq

        acc_partial = tl.zeros([BLOCK_D], dtype=tl.float32)

        for t_start in range(0, T, BLOCK_T):
            t_idx = t_start + tl.arange(0, BLOCK_T)
            mask_t = t_idx < T

            attn_chunk = tl.load(attn_row_ptr + t_idx, mask=mask_t, other=0.0)  # [BLOCK_T]
            Kc_chunk = tl.load(Kc_ptr + t_idx[:, None] * Dq + d_idx[None, :], mask=mask_t[:, None] & mask_d[None, :], other=0.0)  # [BLOCK_T, BLOCK_D]

            acc_partial += tl.sum(attn_chunk[:, None] * Kc_chunk, axis=0)  # [BLOCK_D]

        tl.store(out_row_ptr + d_idx, acc_partial, mask=mask_d)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors and float32 for compute
        device = q_nope.device
        use_cuda = device.type == 'cuda'
        if not use_cuda:
            device = torch.device('cuda')
            q_nope = q_nope.to(device)
            q_pe = q_pe.to(device)
            ckv_cache = ckv_cache.to(device)
            kpe_cache = kpe_cache.to(device)
            kv_indptr = kv_indptr.to(device)
            kv_indices = kv_indices.to(device)

        q_nope_f32 = q_nope.contiguous().to(torch.float32)
        q_pe_f32 = q_pe.contiguous().to(torch.float32)
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 64]

        batch_size = q_nope_f32.shape[0]
        H = q_nope_f32.shape[1]
        Dq = q_nope_f32.shape[2]
        Dp = q_pe_f32.shape[2]

        # Output buffers
        output = torch.empty((batch_size, H, Dq), dtype=torch.float32, device=device)  # compute in fp32, cast later
        lse = torch.empty((batch_size, H), dtype=torch.float32, device=device)
        # attn buffer: [batch, H, T]; we'll compute per-batch
        attn = torch.empty((batch_size, H, 0), dtype=torch.float32, device=device)  # placeholder, will allocate per-batch below

        for b in range(batch_size):
            # Compute range from kv_indptr
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No valid tokens for this batch element; output zeros and skip
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather tokens used
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.int32)
            L_tokens = tok_idx.numel()
            Kc = Kc_all[tok_idx]  # [L_tokens, Dq]
            Kp = Kp_all[tok_idx]  # [L_tokens, Dp]

            # Prepare q vectors for head h=0..H-1; but kernels use q per head h. We'll run one program per head.
            # Allocate per-head outputs
            attn_b = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)

            # Launch fused_logits kernel: one program per head
            grid_logits = (H,)
            fused_logits_kernel[grid_logits](
                q_nope_f32[b], q_pe_f32[b], Kc, Kp, logits,
                T=L_tokens,
                BLOCK_T=256,
            )

            # scaled_logits
            scaled_logits = logits * sm_scale

            # Launch softmax kernel per head
            grid_softmax = (H,)
            attn_b.copy_(torch.empty((H, L_tokens), dtype=torch.float32, device=device))  # re-init
            softmax_row_kernel[grid_softmax](
                scaled_logits, attn_b,
                T=L_tokens,
                BLOCK_T=256,
            )

            # Launch lse kernel per head
            grid_lse = (H,)
            lse_row_kernel[grid_lse](
                scaled_logits, lse[b],
                T=L_tokens,
                BLOCK_T=256,
            )

            # Compute out[h, :] = attn_b[h, :] @ Kc[:, :]
            out_b = torch.empty((H, Dq), dtype=torch.float32, device=device)
            grid_matmul = (H,)
            attn_matmul_kernel_fixed[grid_matmul](
                attn_b, Kc, out_b,
                T=L_tokens, Dq=Dq,
                BLOCK_T=256, BLOCK_D=128,
            )

            # Store results
            output[b] = out_b
            # lse already computed per head in lse[b, h]

        # Cast output to bfloat16 to match original dtype
        output = output.to(torch.bfloat16)
        return output, lse

# The following ModelNew.forward is the entry point required by the evaluation harness.
class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        return self.forward_impl(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)

# Helper to provide forward_impl for entry point
class _ForwardWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.impl = ModelNew()

    def forward(self, *args):
        return self.impl.forward(*args)

# Provide a single entry point as requested
model = _ForwardWrapper()


def run(*args):
    return ModelNew()(*args)
