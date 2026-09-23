import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: gather selected rows from global cache into contiguous per-batch buffers
@triton.jit
def gather_tokens_kernel(
    K_src_ptr,        # *f32, flattened [num_pages, head_dim]
    idx_ptr,          # *i32, [L_tokens] token indices
    out_ptr,          # *f32, contiguous [L_tokens, head_dim]
    num_pages: tl.constexpr,  # int
    head_dim: tl.constexpr,   # int
    L_tokens: tl.constexpr,   # int
):
    pid = tl.program_id(0)  # row id in [0, L_tokens)
    offs = tl.arange(0, head_dim)
    tok_id = tl.load(idx_ptr + pid)
    src_linear = tok_id * head_dim + offs
    vals = tl.load(K_src_ptr + src_linear)
    tl.store(out_ptr + pid * head_dim + offs, vals)


# Triton kernel: compute per-head logits = qn @ K.T + qp @ P.T, write to logits_ptr
@triton.jit
def forward_attention_kernel(
    qn_ptr,           # *f32, [head_dim]
    qp_ptr,           # *f32, [head_dim_kpe]
    K_ptr,            # *f32, [L_tokens, head_dim]
    P_ptr,            # *f32, [L_tokens, head_dim_kpe]
    logits_ptr,       # *f32, [L_tokens]
    head_dim: tl.constexpr,
    head_dim_kpe: tl.constexpr,
    L_tokens: tl.constexpr,
):
    # Compute qn @ K.T -> [L_tokens]
    offs = tl.arange(0, head_dim)
    acc = tl.zeros((L_tokens,), dtype=tl.float32)
    # Loop over k dimension
    for k in range(0, head_dim):
        qk = tl.load(qn_ptr + k)
        col_k = tl.load(K_ptr + offs * head_dim + k)  # vector over tokens
        acc += col_k * qk
    # Compute qp @ P.T -> [L_tokens]
    offs_kpe = tl.arange(0, head_dim_kpe)
    acc2 = tl.zeros((L_tokens,), dtype=tl.float32)
    for k in range(0, head_dim_kpe):
        pk = tl.load(qp_ptr + k)
        col_p = tl.load(P_ptr + offs_kpe * head_dim_kpe + k)  # vector over tokens
        acc2 += col_p * pk
    tl.store(logits_ptr, acc + acc2)


# Triton kernel: softmax over a vector logits_ptr of length L_tokens, write to out_ptr
@triton.jit
def softmax_kernel(
    logits_ptr,       # *f32, [L_tokens]
    out_ptr,          # *f32, [L_tokens]
    L_tokens: tl.constexpr,
    sm_scale: tl.float32,
):
    # Load vector
    offs = tl.arange(0, L_tokens)
    x = tl.load(logits_ptr + offs)
    # Subtract max for numerical stability
    m = tl.max(x, axis=0)
    x = x - m
    x = x * sm_scale
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    out = exp_x / sum_exp
    tl.store(out_ptr + offs, out)


# Triton kernel: matvec out = attn @ K, where attn is [L_tokens], K is [L_tokens, head_dim], out is [head_dim]
@triton.jit
def matvec_kernel(
    attn_ptr,         # *f32, [L_tokens]
    K_ptr,            # *f32, [L_tokens, head_dim]
    out_ptr,          # *f32, [head_dim]
    head_dim: tl.constexpr,
    L_tokens: tl.constexpr,
):
    offs = tl.arange(0, head_dim)
    acc = tl.zeros((head_dim,), dtype=tl.float32)
    # Loop over tokens
    for i in range(0, L_tokens):
        ai = tl.load(attn_ptr + i)
        row_i = tl.load(K_ptr + i * head_dim + offs)
        acc += row_i * ai
    tl.store(out_ptr + offs, acc)


# Triton kernel: compute per-head logsumexp of a vector logits_ptr of length L_tokens
@triton.jit
def lse_per_head_kernel(
    logits_ptr,       # *f32, [L_tokens]
    out_ptr,          # *f32, scalar output (unused length)
    L_tokens: tl.constexpr,
):
    offs = tl.arange(0, L_tokens)
    x = tl.load(logits_ptr + offs)
    m = tl.max(x, axis=0)
    sum_exp = tl.sum(tl.exp(x - m), axis=0)
    lse = m + tl.log(sum_exp)
    # store as scalar at out_ptr (single element)
    tl.store(out_ptr, lse)


# Triton kernel: reduce per-batch lse across heads and divide by num_heads
@triton.jit
def lse_reduce_kernel(
    lse_ptrs,         # *f32, [batch_size, num_qo_heads]
    out_ptr,          # *f32, [batch_size]
    num_heads: tl.constexpr,
):
    b = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)
    for h in range(0, num_heads):
        acc += tl.load(lse_ptrs + b * num_heads + h)
    avg = acc / num_heads
    tl.store(out_ptr + b, avg)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # constants assumed by original code
        self.num_qo_heads = 16
        self.head_dim_ckv = 512
        self.head_dim_kpe = 64
        self.page_size = 1

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors if Triton available
        device = q_nope.device
        assert TRITON_AVAILABLE, "Triton is not available"

        batch_size = q_nope.shape[0]
        num_qo_heads = self.num_qo_heads
        head_dim_ckv = self.head_dim_ckv
        head_dim_kpe = self.head_dim_kpe

        # Prepare per-batch L_tokens and indices
        # kv_indptr: [batch_size + 1], int32
        # tok_idx: [L_tokens] int32
        L_tokens_list = (kv_indptr[1:] - kv_indptr[0]).tolist()
        # Since batch_size == len(kv_indptr) - 1, we can build tok_idx per b
        tok_idx = []
        for b in range(batch_size):
            L_tokens = L_tokens_list[b]
            # gather indices for this batch
            if L_tokens <= 0:
                # No KV cache for this batch element; return zeros output, empty lse
                output = torch.zeros((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
                lse_base2 = torch.empty((batch_size,), dtype=torch.float32, device=device)
                return output, lse_base2
            # indices are a subset of kv_indices specified by kv_indptr
            # Extract range: start at b, end at b+1
            # Note: kv_indptr[b+1] is global index range start for this batch; we need to compute local indices
            # In the original code, kv_indptr[b+1] - kv_indptr[b] == L_tokens, and tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b+1]]
            # Here, kv_indices is a flat list; we must map global indices. However, we cannot access Python lists in Triton,
            # so we pre-gather Kc/Kp in Triton using idx_ptr below.
            tok_idx.append(kv_indices[kv_indptr[b]: kv_indptr[b + 1]])
        # For Triton gather, we only need tok_idx[b] per b. We will pass idx_ptr as a temporary 1D tensor for each b.
        # But gather_tokens_kernel expects a single idx_ptr. We'll gather per b by launching with current batch b inside host loop.

        # Allocate output and per-batch lse tensors
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse_per_head = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)
        lse_base2 = torch.empty((batch_size,), dtype=torch.float32, device=device)

        # Flatten K caches for Triton
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, head_dim_ckv]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, head_dim_kpe]

        for b in range(batch_size):
            L_tokens = L_tokens_list[b]
            # Prepare idx_ptr: global indices for this batch
            idx_ptr = tok_idx[b].to(torch.int32).contiguous()  # [L_tokens]
            # Gather Kc_tmp and Kp_tmp per batch
            Kc_tmp = torch.empty((L_tokens, head_dim_ckv), dtype=torch.float32, device=device)
            Kp_tmp = torch.empty((L_tokens, head_dim_kpe), dtype=torch.float32, device=device)

            # Launch gather kernels: one program per row
            gather_tokens_kernel[(L_tokens,)](
                K_src_ptr=Kc_all,
                idx_ptr=idx_ptr,
                out_ptr=Kc_tmp,
                num_pages=Kc_all.shape[0],
                head_dim=head_dim_ckv,
                L_tokens=L_tokens,
            )
            gather_tokens_kernel[(L_tokens,)](
                K_src_ptr=Kp_all,
                idx_ptr=idx_ptr,
                out_ptr=Kp_tmp,
                num_pages=Kp_all.shape[0],
                head_dim=head_dim_kpe,
                L_tokens=L_tokens,
            )

            # Compute output and per-head lse per head
            for h in range(num_qo_heads):
                # qn and qp
                qn = q_nope[b, h].to(torch.float32).contiguous()  # [head_dim_ckv]
                qp = q_pe[b, h].to(torch.float32).contiguous()   # [head_dim_kpe]

                # Logits: [L_tokens]
                logits = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                forward_attention_kernel[(1,)](
                    qn_ptr=qn,
                    qp_ptr=qp,
                    K_ptr=Kc_tmp,
                    P_ptr=Kp_tmp,
                    logits_ptr=logits,
                    head_dim=head_dim_ckv,
                    head_dim_kpe=head_dim_kpe,
                    L_tokens=L_tokens,
                    sm_scale=float(sm_scale),
                )

                # Softmax over logits_scaled
                attn = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                softmax_kernel[(L_tokens,)](
                    logits_ptr=logits,
                    out_ptr=attn,
                    L_tokens=L_tokens,
                    sm_scale=float(sm_scale),
                )

                # Matvec: output[h] = attn @ Kc_tmp -> [head_dim_ckv]
                out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
                matvec_kernel[(head_dim_ckv,)](
                    attn_ptr=attn,
                    K_ptr=Kc_tmp,
                    out_ptr=out_vec,
                    head_dim=head_dim_ckv,
                    L_tokens=L_tokens,
                )
                output[b, h] = out_vec

                # Per-head lse: logsumexp of logits * sm_scale
                lse_val = torch.empty((), dtype=torch.float32, device=device)
                lse_per_head_kernel[(L_tokens,)](
                    logits_ptr=logits,
                    out_ptr=lse_val,
                    L_tokens=L_tokens,
                )
                lse_per_head[b, h] = lse_val

        # Reduce across heads and convert to base-2 (original divides by ln(2))
        lse_reduce_kernel[(batch_size,)](
            lse_ptrs=lse_per_head,
            out_ptr=lse_base2,
            num_heads=num_qo_heads,
        )

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)

        return output, lse_base2


def run(*args):
    return ModelNew()(*args)
