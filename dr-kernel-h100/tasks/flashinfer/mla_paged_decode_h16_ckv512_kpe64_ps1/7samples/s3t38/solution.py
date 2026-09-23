import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels
if TRITON_AVAILABLE:
    @triton.jit
    def compute_logits_kernel_bh(qn_ptr, Kc_ptr, qp_ptr, Kp_ptr, logits_ptr,
                                 L_b: tl.constexpr, Dc: tl.constexpr, Dp: tl.constexpr,
                                 sm_scale: tl.float32):
        # One Triton program per (b, h), but we pass grid=(1,1) and decode b,h via pid
        # Note: Triton grid is used here as (1,1) to avoid complicated indexing; host handles loops over b,h.
        b = tl.program_id(0)
        h = tl.program_id(1)
        # Load qn[h, :] and qp[h, :]
        qn = tl.load(qn_ptr)  # [Dc]
        qp = tl.load(qp_ptr)  # [Dp]

        # Compute logits_scaled[l] = sm_scale * (qn @ Kc[l, :] + qp @ Kp[l, :]) for l in [0..L_b-1]
        for l in tl.static_range(0, L_b):
            acc = 0.0
            for k in tl.static_range(0, Dc):
                acc += qn[k] * tl.load(Kc_ptr + l * Dc + k)
            for k in tl.static_range(0, Dp):
                acc += qp[k] * tl.load(Kp_ptr + l * Dp + k)
            acc = acc * sm_scale
            tl.store(logits_ptr + (b * H + h) * L_b + l, acc)

    @triton.jit
    def compute_lse_kernel_bh(logits_ptr, lse_ptr,
                              L_b: tl.constexpr):
        b = tl.program_id(0)
        h = tl.program_id(1)
        # Load logits vector for (b, h)
        max_val = -float('inf')
        for l in tl.static_range(0, L_b):
            val = tl.load(logits_ptr + (b * H + h) * L_b + l)
            max_val = tl.maximum(max_val, val)
        sumexp = 0.0
        for l in tl.static_range(0, L_b):
            val = tl.load(logits_ptr + (b * H + h) * L_b + l)
            sumexp += tl.exp(val - max_val)
        lse = tl.log(sumexp) + max_val  # base-e logsumexp
        # Store as float32
        tl.store(lse_ptr + 0, lse)

    @triton.jit
    def compute_softmax_kernel_bh(logits_ptr, attn_ptr,
                                  L_b: tl.constexpr):
        b = tl.program_id(0)
        h = tl.program_id(1)
        max_val = -float('inf')
        for l in tl.static_range(0, L_b):
            val = tl.load(logits_ptr + (b * H + h) * L_b + l)
            max_val = tl.maximum(max_val, val)
        sumexp = 0.0
        for l in tl.static_range(0, L_b):
            val = tl.load(logits_ptr + (b * H + h) * L_b + l)
            sumexp += tl.exp(val - max_val)
        for l in tl.static_range(0, L_b):
            val = tl.load(logits_ptr + (b * H + h) * L_b + l)
            attn = tl.exp(val - max_val) / sumexp
            tl.store(attn_ptr + l, attn)

    @triton.jit
    def compute_out_kernel_bh(attn_ptr, Kc_ptr, out_ptr,
                              Dc: tl.constexpr, L_b: tl.constexpr):
        # out = attn @ Kc, reducing over L_b
        b = tl.program_id(0)
        h = tl.program_id(1)
        for k in tl.static_range(0, Dc):
            acc = 0.0
            for l in tl.static_range(0, L_b):
                attn = tl.load(attn_ptr + l)
                val = tl.load(Kc_ptr + l * Dc + k)
                acc += attn * val
            tl.store(out_ptr + (b * H + h) * Dc + k, acc)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Ensure device is CUDA if Triton available
    device = q_nope.device
    B, H, Dc = q_nope.shape
    Dp = q_pe.shape[-1]
    # We assert fixed constants as in original
    assert num_qo_heads is None, "num_qo_heads not defined at call site; assume H from q_nope"
    # Prepare outputs
    output = torch.empty((B, H, Dc), dtype=torch.float32, device=device)
    lse = torch.empty((B, H), dtype=torch.float32, device=device)

    # Loop over batch and heads; Triton kernels handle per-(b,h) work
    for b in range(B):
        # Derive token indices for this batch element
        L_b = int((kv_indptr[b + 1] - kv_indptr[b]).item())
        tok_idx = kv_indices[kv_indptr[b].item():kv_indptr[b + 1].item()].to(torch.int64)
        # Gather Kc and Kp
        Kc_b = ckv_cache[tok_idx, 0].to(torch.float32).contiguous()  # [L_b, Dc]
        Kp_b = kpe_cache[tok_idx, 0].to(torch.float32).contiguous()  # [L_b, Dp]

        # For each head h
        for h in range(H):
            # Compute logits_scaled vector for this head (length L_b)
            logits_vec = torch.empty(L_b, dtype=torch.float32, device=device)
            # Prepare qn and qp pointers
            qn = q_nope[b, h].to(torch.float32).contiguous()
            qp = q_pe[b, h].to(torch.float32).contiguous()

            # Launch Triton kernel to compute logits_scaled
            compute_logits_kernel_bh[(1, 1)](qn, Kc_b, qp, Kp_b, logits_vec,
                                             L_b=L_b, Dc=Dc, Dp=Dp, sm_scale=float(sm_scale))

            # Compute lse for this head (base-2 logsumexp)
            lse_bh = torch.empty((), dtype=torch.float32, device=device)
            compute_lse_kernel_bh[(1, 1)](logits_vec, lse_bh,
                                          L_b=L_b)
            lse[b, h] = lse_bh.item()

            # Compute softmax attn vector for this head
            attn = torch.empty(L_b, dtype=torch.float32, device=device)
            compute_softmax_kernel_bh[(1, 1)](logits_vec, attn,
                                              L_b=L_b)

            # Compute final output vector: out[h, :] = attn @ Kc_b
            out_vec = torch.empty(Dc, dtype=torch.float32, device=device)
            compute_out_kernel_bh[(1, 1)](attn, Kc_b, out_vec,
                                          Dc=Dc, L_b=L_b)
            output[b, h] = out_vec

    # Cast output to bfloat16 to match original
    output = output.to(torch.bfloat16)
    return output, lse


def get_inputs():
    # Example inputs, evaluator provides its own; these are here to demonstrate usage.
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to('cuda')
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32).to('cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
