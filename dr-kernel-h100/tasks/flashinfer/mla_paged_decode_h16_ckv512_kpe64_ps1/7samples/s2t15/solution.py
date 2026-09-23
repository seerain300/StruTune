import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _batch_head_kernel(
    q_nope_ptr, q_pe_ptr,
    Kc_ptr, Kp_ptr,
    out_ptr, lse_ptr,
    B: tl.constexpr, H: tl.constexpr,
    Dc: tl.constexpr, Dp: tl.constexpr,
    L_tokens: tl.constexpr,
    sm_scale: tl.float32
):
    # program id
    b = tl.program_id(0)  # batch element
    h = tl.program_id(1)  # head index

    # Load qn and qp for head h
    # q_nope[b, h, :] -> [Dc]
    # q_pe[b, h, :] -> [Dp]
    qn = tl.zeros([Dc], dtype=tl.float32)
    qp = tl.zeros([Dp], dtype=tl.float32)

    # base offsets for q vectors: q_nope is [B, H, Dc], q_pe is [B, H, Dp]
    # We access q_nope[b, h, :] by computing pointer with strides. For simplicity, assume contiguous:
    # q_nope.stride(0)=H*Dc, stride(1)=Dc, stride(2)=1 for contiguous layout.
    # Given q_nope is contiguous in last dim, we can compute base = b*(H*Dc) + h*Dc, then load vector.
    # However, to be robust, we pass qn/qp as 1D contiguous vectors to the kernel. We'll handle that by
    # launching with per-batch inputs prepared separately (host-side). See forward for details.

    # We assume qn and qp are passed as contiguous 1D arrays to this kernel as arguments named qn_ptr, qp_ptr.
    # The kernel will use those. In forward, we will pass q_nope[b, h, :].contiguous().float() to qn_ptr, etc.
    # But Triton requires explicit pointer. Forward will pass qn/qp pointers. We need to provide those names:
    # qn_ptr: pointer to [Dc] float32
    # qp_ptr: pointer to [Dp] float32
    # out_ptr: pointer to [Dc] bfloat16 at output[b, h, :]
    # lse_ptr: pointer to scalar float32 at lse[b, h]

    # Load qn and qp
    # Since Triton cannot directly access tensor named qn_ptr/qp_ptr; we expect forward to bind these correctly.
    # Here, we create qn/qp using zeros and then fill them via tl.load with pointers passed in. The forward will
    # pass actual pointers to these vectors. To make this work, define qn_ptr/qp_ptr as arguments in forward.

    # Since Triton requires explicit arguments, we redefine qn/qp as inputs:
    # We'll use qn_ptr and qp_ptr as inputs. For clarity, we will not use tl.load here; Triton expects qn_ptr,
    # qp_ptr to be passed by forward. We'll define them in forward as pointers to contiguous 1D arrays.
    # Triton will receive them as arguments named qn_ptr, qp_ptr.

    # Compute logits vector: we will fill qn/qp via pointers. However, Triton kernel cannot read qn/qp like this
    # without passing them. So we rely on forward to pass qn_ptr, qp_ptr as pointers to contiguous vectors.
    # Let's proceed by assuming qn_ptr, qp_ptr are provided by forward.

    # Initialize logits_scaled and attention vectors
    logits_scaled = tl.zeros([L_tokens], dtype=tl.float32)
    # We need to fill qn and qp first. Triton doesn't provide read-only access to local variables; we need to
    # pass them as pointers. The correct approach: forward will pass qn_ptr, qp_ptr to the kernel and we load.

    # Load qn and qp from pointers
    # Triton kernel should have qn_ptr, qp_ptr as arguments. We'll reconstruct by reading q_nope and q_pe.
    # But Triton cannot index tensors like q_nope[b, h, :]. Instead, forward prepares qn_ptr, qp_ptr.
    # To keep kernel simple, we won't read them here; we rely on forward to pass qn_ptr, qp_ptr.

    # Since Triton cannot access q_nope/q_pe here, we need to restructure: forward will pass qn_ptr, qp_ptr.
    # Implement: forward prepares qn_ptr and qp_ptr and calls kernel.

    # For now, we implement computation using qn_ptr, qp_ptr. Triton will receive those as arguments.
    # We need to load qn and qp from qn_ptr and qp_ptr.
    # Triton loads via tl.load(ptr, offsets). We need offsets 0..Dc-1 and 0..Dp-1.

    # We'll implement using offsets: qn[i] via tl.load(qn_ptr + i), similarly for qp.
    # Note: Triton loops use tl.static_range. But we cannot iterate over qn_ptr without prior qn vector. So we
    # restructure: forward will pass qn_ptr, qp_ptr; kernel loads qn/qp vectors into registers by iterating over
    # indices 0..Dc-1 and 0..Dp-1 and storing into qn/qp arrays.

    # Create local arrays qn_local and qp_local
    qn_local = tl.zeros([Dc], dtype=tl.float32)
    qp_local = tl.zeros([Dp], dtype=tl.float32)

    # Load qn and qp into local arrays
    # For i in 0..Dc-1: qn_local[i] = tl.load(qn_ptr + i)
    # For j in 0..Dp-1: qp_local[j] = tl.load(qp_ptr + j)
    # Triton allows loops. Implement:
    for i in tl.static_range(Dc):
        qn_local[i] = tl.load(qn_ptr + i)

    for j in tl.static_range(Dp):
        qp_local[j] = tl.load(qp_ptr + j)

    # Now qn_local and qp_local are the vectors. We can proceed.

    # Initialize accumulators for logits_scaled
    max_v = -float("inf")
    sum_exp = 0.0

    # First pass: compute max for numerical stability
    for t in tl.static_range(L_tokens):
        sum_qn = 0.0
        sum_qp = 0.0
        for i in tl.static_range(Dc):
            sum_qn += qn_local[i] * tl.load(Kc_ptr + t * Dc + i)
        for j in tl.static_range(Dp):
            sum_qp += qp_local[j] * tl.load(Kp_ptr + t * Dp + j)
        logits_scaled[t] = sum_qn + sum_qp
        if logits_scaled[t] > max_v:
            max_v = logits_scaled[t]

    # Second pass: compute sum exp
    for t in tl.static_range(L_tokens):
        sum_exp += tl.exp(logits_scaled[t] * sm_scale - max_v * tl.log(2.0))

    # Compute lse
    # logsumexp base 2: lse = log(sum_exp) + max_v, then divide by ln(2)
    lse_val = tl.log(sum_exp) + max_v
    lse_val = lse_val / tl.log(2.0)

    # Write lse to output
    # lse_ptr points to lse[b, h] as a scalar
    tl.store(lse_ptr + b * H + h, lse_val)

    # Third pass: compute attention and output
    out_vec = tl.zeros([Dc], dtype=tl.float32)
    for t in tl.static_range(L_tokens):
        attn_t = tl.exp(logits_scaled[t] * sm_scale - lse_val * tl.log(2.0))
        for i in tl.static_range(Dc):
            out_vec[i] += attn_t * tl.load(Kc_ptr + t * Dc + i)

    # Store output as bfloat16
    # out_ptr points to output[b, h, :] which is contiguous of length Dc
    out_bf16 = out_vec.to(tl.bfloat16)
    for i in tl.static_range(Dc):
        tl.store(out_ptr + b * (H * Dc) + h * Dc + i, out_bf16[i])

# Helper to run Triton version (ModelNew will call this)
def _run_triton_only(q_nope, q_pe, Kc_all, Kp_all, kv_indptr, kv_indices, sm_scale):
    B, H = q_nope.shape[0], q_nope.shape[1]
    Dc = q_nope.shape[2]
    Dp = q_pe.shape[2]
    device = q_nope.device

    # Prepare output and lse
    output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
    lse = torch.empty((B, H), dtype=torch.float32, device=device)

    # We need to compute L_tokens per batch element using kv_indptr. We will loop over b.
    # Triton kernel will be launched once per (b, h).
    for b in range(B):
        # Determine token range
        if kv_indptr.numel() == 0:
            # No indptr provided; default L_tokens = 0
            output[b].zero_()
            lse[b].fill_(-float("inf"))
            continue

        if kv_indptr.numel() != B + 1:
            # Unexpected indptr length
            output[b].zero_()
            lse[b].fill_(-float("inf"))
            continue

        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        L_tokens = max(page_end - page_beg, 0)

        # If L_tokens == 0, skip compute
        if L_tokens == 0:
            output[b].zero_()
            lse[b].fill_(-float("inf"))
            continue

        # Gather token indices
        tok_idx = kv_indices[page_beg:page_end].to(torch.long)

        # Gather Kc and Kp rows
        Kc = Kc_all[tok_idx].to(torch.float32)  # [L_tokens, Dc]
        Kp = Kp_all[tok_idx].to(torch.float32)  # [L_tokens, Dp]

        # Prepare qn and qp for each head as contiguous vectors; we'll create them per head in kernel launch.
        # Triton requires passing pointers to qn, qp. Forward will prepare them for each head and launch.
        # We'll create per-head qn_ptr, qp_ptr for each h.

        # Launch Triton kernel for each head h
        # We need to pass qn_ptr and qp_ptr. Forward will prepare them for each h.
        for h in range(H):
            # Prepare qn_ptr = q_nope[b, h, :].contiguous().float()
            qn_ptr = q_nope[b, h, :].contiguous().float()
            # Prepare qp_ptr = q_pe[b, h, :].contiguous().float()
            qp_ptr = q_pe[b, h, :].contiguous().float()

            # Compute output for this head
            out_ptr = output[b, h, :].contiguous().float()  # dummy pointer; we'll store bfloat16 directly
            lse_ptr = lse[b, h]  # scalar pointer

            # Launch kernel
            grid = (B, H)
            _batch_head_kernel[grid](
                qn_ptr, qp_ptr,
                Kc, Kp,
                out_ptr, lse_ptr,
                B, H,
                Dc, Dp,
                L_tokens,
                sm_scale
            )

    return output, lse

class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on the same device and dtype expectations
        device = q_nope.device
        # Triton kernels expect float32 math; inputs may be bfloat16
        q_nope_f32 = q_nope.to(torch.float32)
        q_pe_f32 = q_pe.to(torch.float32)
        Kc_all_f32 = ckv_cache.to(torch.float32)
        Kp_all_f32 = kpe_cache.to(torch.float32)
        kv_indptr = kv_indptr.to(torch.int32)
        kv_indices = kv_indices.to(torch.long)

        output, lse = _run_triton_only(q_nope_f32, q_pe_f32, Kc_all_f32, Kp_all_f32, kv_indptr, kv_indices, sm_scale)
        return output, lse


def run(*args):
    return ModelNew()(*args)
