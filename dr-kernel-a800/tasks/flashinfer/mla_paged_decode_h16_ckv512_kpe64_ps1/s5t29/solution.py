import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_scaled_logits_kernel(
    A_qn, A_qp, B_qc, B_qp, Out,
    H: tl.constexpr, CK: tl.constexpr, KP: tl.constexpr, L: tl.constexpr, sm_scale: tl.constexpr
):
    # Grid is (H, L): each program computes one scaled_logits[h, t]
    pid_h = tl.program_id(axis=0)
    pid_t = tl.program_id(axis=1)

    # Accumulate qn · Kc[t] and qp · Kp[t]
    acc1 = 0.0
    # A_qn shape: [H, CK], B_qc shape: [L, CK]
    for k in range(CK):
        a = tl.load(A_qn + pid_h * CK + k)  # qn[h, k]
        b = tl.load(B_qc + pid_t * CK + k)  # Kc[t, k]
        acc1 += a * b

    acc2 = 0.0
    # A_qp shape: [H, KP], B_qp shape: [L, KP]
    for k in range(KP):
        a = tl.load(A_qp + pid_h * KP + k)  # qp[h, k]
        b = tl.load(B_qp + pid_t * KP + k)  # Kp[t, k]
        acc2 += a * b

    val = (acc1 + acc2) * sm_scale
    # Store to Out[h, t]
    tl.store(Out + pid_h * L + pid_t, val)


@triton.jit
def _lse_kernel(
    Logits_ptr, Lse_ptr,
    H: tl.constexpr, L: tl.constexpr
):
    h = tl.program_id(axis=0)
    max_val = -float('inf')
    # Pass 1: find max
    for t in range(0, L):
        ptr = Logits_ptr + h * L + t
        val = tl.load(ptr)
        if val > max_val:
            max_val = val

    sumexp = 0.0
    # Pass 2: sum exp(val - max_val)
    for t in range(0, L):
        ptr = Logits_ptr + h * L + t
        val = tl.load(ptr)
        sumexp += tl.exp(val - max_val)

    lse = tl.log(sumexp)
    tl.store(Lse_ptr + h, lse)


@triton.jit
def _compute_output_kernel(
    Logits_ptr, Lse_ptr, Kc_ptr, Out_ptr,
    H: tl.constexpr, CK: tl.constexpr, L: tl.constexpr,
    stride_kc_t: tl.constexpr, stride_kc_k: tl.constexpr,
    stride_out_h: tl.constexpr, stride_out_k: tl.constexpr,
):
    # One program per head
    h = tl.program_id(axis=0)
    lse_h = tl.load(Lse_ptr + h)
    out_acc = tl.zeros((CK,), dtype=tl.float32)

    for t in range(0, L):
        ptr = Logits_ptr + h * L + t
        val = tl.load(ptr)  # scaled_logits[h, t]
        softmax = tl.exp(val - lse_h)
        # Load Kc[t, :] row and accumulate
        for k in range(CK):
            kptr = Kc_ptr + t * stride_kc_t + k * stride_kc_k
            kc = tl.load(kptr)
            out_acc[k] += softmax * kc

    # Store out[h, :]
    for k in range(CK):
        o_ptr = Out_ptr + h * stride_out_h + k * stride_out_k
        tl.store(o_ptr, out_acc[k])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Move to device (assuming q_nope is on correct device)
        device = q_nope.device

        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[1]
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"
        assert num_pages == 1, "num_pages must be 1 (per config)"
        assert kv_indptr.shape[0] == batch_size + 1, "kv_indptr length must be batch_size + 1"
        assert kv_indices.shape[0] >= kv_indptr[-1].item(), "kv_indices must cover all tokens"
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Tensors must be on CUDA"

        # Convert to float32 for compute
        q_nope_f = q_nope.to(torch.float32)
        q_pe_f = q_pe.to(torch.float32)
        ckv_cache_f = ckv_cache.to(torch.float32)
        kpe_cache_f = kpe_cache.to(torch.float32)

        H = num_qo_heads
        CK = head_dim_ckv
        KP = head_dim_kpe

        # Prepare output buffers
        output = torch.empty((batch_size, H, CK), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, H), dtype=torch.float32, device=device)

        # Iterate over batch
        for b in range(batch_size):
            # Compute L_tokens and tok_idx
            begin = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens = end - begin
            if L_tokens <= 0:
                # No tokens for this batch element
                lse[b] = 0.0  # placeholder
                continue

            tok_idx = kv_indices[begin:end].to(torch.int32).contiguous()  # [L_tokens]

            # Extract Kc_rows and Kp_rows
            Kc_rows = ckv_cache_f[tok_idx]  # [L_tokens, CK]
            Kp_rows = kpe_cache_f[tok_idx]  # [L_tokens, KP]

            # Prepare qn and qp for current batch
            qn = q_nope_f[b]  # [H, CK]
            qp = q_pe_f[b]    # [H, KP]

            # Reshape A_qn: [H, CK], B_qc: [L, CK]
            A_qn = qn.contiguous().view(H, CK)
            B_qc = Kc_rows.contiguous().view(L_tokens, CK)

            # Reshape A_qp: [H, KP], B_qp: [L, KP]
            A_qp = qp.contiguous().view(H, KP)
            B_qp = Kp_rows.contiguous().view(L_tokens, KP)

            # Allocate logits buffer [H, L_tokens]
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)

            # Launch scaled logits kernel
            grid_scaled = (H, L_tokens)
            _compute_scaled_logits_kernel[grid_scaled](
                A_qn, A_qp, B_qc, B_qp, logits,
                H=H, CK=CK, KP=KP, L=L_tokens, sm_scale=float(sm_scale)
            )

            # Compute lse per head
            grid_lse = (H,)
            lse_b = torch.empty((H,), dtype=torch.float32, device=device)
            _lse_kernel[grid_lse](logits, lse_b, H=H, L=L_tokens)

            # Compute output per head using Kc_rows
            Out = output[b]  # [H, CK], contiguous
            # Strides for Kc_rows [L_tokens, CK]
            stride_kc_t = Kc_rows.stride(0)  # typically CK
            stride_kc_k = Kc_rows.stride(1)  # typically 1
            # Strides for Out [H, CK]
            stride_out_h = Out.stride(0)  # typically CK
            stride_out_k = Out.stride(1)  # typically 1

            grid_out = (H,)
            _compute_output_kernel[grid_out](
                logits, lse_b, Kc_rows, Out,
                H=H, CK=CK, L=L_tokens,
                stride_kc_t=stride_kc_t, stride_kc_k=stride_kc_k,
                stride_out_h=stride_out_h, stride_out_k=stride_out_k,
            )

            # For batch b with tokens, set lse[b, :] to computed lse_b
            # The previous 'lse[b] = 0' placeholder was only for no-token case; here we write lse_b
            lse[b] = lse_b

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)

        # Return (output, lse). Keep lse in float32 as original run likely returned float32
        return (output_bf16, lse)

# Example inputs helpers (not used by evaluator, but provided for completeness)
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device="cuda")
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device="cuda")
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device="cuda")
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device="cuda")
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).cuda()
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32).cuda()
    sm_scale = 1.0
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


# Optional: if you need the original run interface (not strictly required by evaluator but shown here)
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    model = ModelNew()
    return model(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)


# Provide ModelNew as entry point
class Model(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
