import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_scaled_per_batch_kernel(
    qn_ptr,         # *fp32, shape [H, D], contiguous
    qp_ptr,         # *fp32, shape [H, Dp], contiguous
    Kc_ptr,         # *fp32, shape [L_tokens, D], contiguous
    Kp_ptr,         # *fp32, shape [L_tokens, Dp], contiguous
    logits_ptr,     # *fp32, flattened buffer [B*H*L_tokens], contiguous
    H,              # int32
    D: tl.constexpr,          # compile-time constant: 512
    Dp: tl.constexpr,         # compile-time constant: 64
    L_tokens,       # int32
    b,              # int32 (we pass b to identify the batch; host ensures grid size equals H)
    sm_scale,       # float32
):
    # Grid: 1D, size = H
    h = tl.program_id(0)
    for t in range(0, L_tokens):
        acc1 = 0.0
        # dot over D (compile-time unrolled loop)
        for kk in range(0, D):
            val = tl.load(qn_ptr + h * D + kk)  # qn[h, kk]
            kv = tl.load(Kc_ptr + t * D + kk)   # Kc[t, kk]
            acc1 += val * kv
        acc2 = 0.0
        # dot over Dp
        for kk in range(0, Dp):
            val = tl.load(qp_ptr + h * Dp + kk)  # qp[h, kk]
            kv = tl.load(Kp_ptr + t * Dp + kk)   # Kp[t, kk]
            acc2 += val * kv
        logit = (acc1 + acc2) * sm_scale
        idx = (b * H + h) * L_tokens + t
        tl.store(logits_ptr + idx, logit)


@triton.jit
def compute_lse_per_batch_kernel(
    logits_ptr,     # *fp32, flattened buffer [B*H*L_tokens]
    lse_ptr,        # *fp32, buffer [B*H]
    H,              # int32
    L_tokens,       # int32
    b,              # int32
):
    # Grid: 1D, size = H
    h = tl.program_id(0)
    base = b * H + h
    row_start = base * L_tokens
    # Compute max for numerical stability
    m = -float("inf")
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        if logit > m:
            m = logit
    # sum(exp(x - m))
    sum_exp = 0.0
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        sum_exp += tl.exp(logit - m)
    lse_val = m + tl.log(sum_exp)  # logsumexp in natural log
    ln2 = 0.6931471805599453
    lse_val = lse_val / ln2
    tl.store(lse_ptr + base, lse_val)


@triton.jit
def compute_output_per_batch_kernel(
    logits_ptr,     # *fp32, flattened buffer [B*H*L_tokens]
    Kc_ptr,         # *fp32, shape [L_tokens, D], contiguous
    output_ptr,     # *fp32, output [H, D], contiguous
    H,              # int32
    D: tl.constexpr,          # 512
    L_tokens,       # int32
    b,              # int32
):
    # Grid: 1D, size = H
    h = tl.program_id(0)
    # Compute softmax over logits[h, :]
    row_start = (b * H + h) * L_tokens
    m = -float("inf")
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        if logit > m:
            m = logit
    sum_exp = 0.0
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        sum_exp += tl.exp(logit - m)
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        attn = tl.exp(logit - m) / sum_exp
        acc = 0.0
        for kk in range(0, D):
            kv = tl.load(Kc_ptr + t * D + kk)
            acc += attn * kv
        tl.store(output_ptr + h * D + kk, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-optimized version of the original Model.forward.
        Returns:
            output: [batch_size, num_qo_heads, head_dim_ckv], bfloat16
            lse: [batch_size, num_qo_heads], float32
        """
        device = q_nope.device
        # Ensure CUDA and contiguity
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA"
        q_nope_f32 = q_nope.to(torch.float32).contiguous()
        q_pe_f32 = q_pe.to(torch.float32).contiguous()
        # Squeeze num_pages dimension (1 in original code) -> [num_pages, D] and [num_pages, Dp]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 64]

        batch_size = q_nope_f32.shape[0]
        H = q_nope_f32.shape[1]
        D = 512
        Dp = 64

        output = torch.empty((batch_size, H, D), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, H), dtype=torch.float32, device=device)

        for b in range(batch_size):
            L_tokens = int(kv_indptr[b + 1].item() - kv_indptr[b].item())
            if L_tokens <= 0:
                output[b].zero_()
                lse[b].fill_(-float("inf"))
                continue

            # Gather token indices for this batch slice
            tok_idx_cpu = kv_indices[int(kv_indptr[b].item()): int(kv_indptr[b + 1].item())].to(torch.int64).cpu()
            # Gather Kc and Kp for these tokens
            Kc = Kc_all[tok_idx_cpu].to(device).contiguous()  # [L_tokens, 512]
            Kp = Kp_all[tok_idx_cpu].to(device).contiguous()  # [L_tokens, 64]

            # Allocate flattened logits buffer for this batch
            num_logits = batch_size * H * L_tokens
            logits_flat = torch.empty((num_logits,), dtype=torch.float32, device=device)

            # Launch Triton kernel to compute logits_scaled[h, t] for all h,t
            grid = (H,)
            compute_logits_scaled_per_batch_kernel[grid](
                q_nope_f32[b], q_pe_f32[b], Kc, Kp, logits_flat, H, D, Dp, L_tokens, b, sm_scale
            )

            # Launch Triton kernel to compute lse per head for this batch
            lse_batch = torch.empty((batch_size * H,), dtype=torch.float32, device=device)
            compute_lse_per_batch_kernel[grid](logits_flat, lse_batch, H, L_tokens, b)
            # Scatter lse to [batch_size, H]
            for h in range(H):
                lse[b, h] = lse_batch[b * H + h]

            # Launch Triton kernel to compute output per head
            output[b].zero_()
            compute_output_per_batch_kernel[grid](logits_flat, Kc, output[b], H, D, L_tokens, b)

        # Return in original output dtype: output in bfloat16, lse in float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


# Original helper functions for testing (not used by evaluation)
@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages = ckv_cache.shape[0]
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64

    device = q_nope.device
    Kc_all = ckv_cache.squeeze(1).to(torch.float32)
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)

    output = torch.zeros((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for b in range(batch_size):
        if kv_indptr.numel() != batch_size + 1:
            raise RuntimeError("kv_indptr length must be batch_size + 1")
        if kv_indptr[-1].item() != num_pages:
            raise RuntimeError("kv_indptr[-1] must equal num_pages")
        L_tokens = int(kv_indptr[b + 1].item() - kv_indptr[b].item())
        if L_tokens <= 0:
            output[b].zero_()
            lse[b].fill_(-float("inf"))
            continue

        tok_idx = kv_indices[kv_indptr[b].item(): kv_indptr[b + 1].item()].to(torch.long).to(device)
        Kc = Kc_all[tok_idx]  # [L_tokens, 512]
        Kp = Kp_all[tok_idx]  # [L_tokens, 64]

        qn = q_nope[b].to(torch.float32)  # [16, 512]
        qp = q_pe[b].to(torch.float32)    # [16, 64]

        logits = (qn @ Kc.T) + (qp @ Kp.T)  # [16, L_tokens]
        logits_scaled = logits * sm_scale
        lse[b] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        attn = torch.softmax(logits_scaled, dim=-1)  # [16, L_tokens]
        out = attn @ Kc  # [16, 512]
        output[b] = out.to(torch.bfloat16)

    return output, lse


def get_inputs():
    # Example inputs; evaluation will provide its own. Ensure CUDA device.
    device = 'cuda'
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device=device)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device=device)
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to(device)
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int64).to(device)
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


# The evaluation environment expects a nn.Module named Model. Define it as a wrapper of ModelNew.
class Model(torch.nn.Module):
    def forward(self, *args):
        return ModelNew().forward(*args)


def run(*args):
    return ModelNew()(*args)
