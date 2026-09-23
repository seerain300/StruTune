import math
import torch
import triton
import triton.language as tl


# Kernel: extract Q vector Q[q_idx, g, :] into out_ptr[0:D]
@triton.jit
def compute_q_vec_kernel(q_ptr, q_idx, g, out_ptr, D: tl.constexpr):
    base = q_idx * G * D + g * D
    offs = tl.arange(0, D)
    vals = tl.load(q_ptr + base + offs)
    tl.store(out_ptr + offs, vals)


# Kernel: for a single q position (q_idx), compute attention logits for gh=0 and output head g=0,
# store logits in out_logit_ptr[0:N], and write LSE (base-2) to lse_ptr[0].
@triton.jit
def attn_logits_lse_gh0_g0_kernel(
    q_vec_ptr,      # [D], float32
    k_ptrs,         # [N, GH, D], float32
    v_ptrs,         # [N, GH, D], float32
    out_logit_ptr,  # [N], float32
    lse_ptr,        # [1], float32
    q_idx,          # int32
    N,              # int32
    D: tl.constexpr,
    GH: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Single chunk processing: BLOCK_N >= N (we set BLOCK_N = N)
    j = tl.arange(0, N)
    k_row_base = k_ptrs + j * GH * D  # gh=0
    v_row_base = v_ptrs + j * GH * D  # gh=0
    k_chunk = tl.load(k_row_base, mask=True, other=0.0)  # [N, D]
    v_chunk = tl.load(v_row_base, mask=True, other=0.0)  # [N, D]

    q_vec = tl.load(q_vec_ptr)  # [D]
    logits = tl.zeros((N,), dtype=tl.float32)
    for d in tl.static_range(0, D):
        qd = q_vec[d]
        k_col = k_chunk[:, d]  # [N]
        logits += qd * k_col

    # Store logits
    for i in tl.static_range(0, N):
        tl.store(out_logit_ptr + i, logits[i])

    # LSE in base-2
    sumexp = tl.sum(tl.exp(logits))
    lse_val = tl.log(sumexp) / math.log(2.0)
    tl.store(lse_ptr, lse_val)


# Kernel: softmax over logits and produce output for output head g (for gh=0),
# stores output into out_ptr[q_idx, g, :]. Assumes q_idx and g known at launch.
@triton.jit
def softmax_and_store_out_kernel(
    logits_ptr,   # [N], float32
    v_ptrs,       # [N, GH, D], float32
    out_ptr,      # [M, G, D], float32 (we write a single element per launch)
    q_idx,        # int32
    g,            # int32
    N,            # int32
    D: tl.constexpr,
    GH: tl.constexpr,
):
    j = tl.arange(0, N)
    v_row_base = v_ptrs + j * GH * D  # gh=0
    logits = tl.load(logits_ptr + j)  # [N]
    # Causal mask: j < q_idx + 1
    mask = j < (q_idx + 1)
    logits = tl.where(mask, logits, -float('inf'))

    # Softmax
    max_logit = tl.max(logits, axis=0)
    logits = logits - max_logit
    exp_logits = tl.exp(logits)
    sumexp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sumexp  # [N]

    # Output QH[g, :] = sum_j probs[j] * V[j, gh=0, :]
    out_base = (q_idx * G + g) * D
    for d in tl.static_range(0, D):
        acc = tl.zeros((), dtype=tl.float32)
        for jj in tl.static_range(0, N):
            v_elem = tl.load(v_ptrs + jj * GH * D + d)  # V[jj, 0, d]
            acc += probs[jj] * v_elem
        tl.store(out_ptr + out_base + d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.G = 32
        self.GH = 8
        self.D = 128
        self.sm_scale = 1.0 / math.sqrt(128)  # original scaling

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Validate shapes and device
        assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
        assert q.shape[1] == self.G and k.shape[1] == self.GH and v.shape[1] == self.GH
        assert q.shape[2] == self.D and k.shape[2] == self.D and v.shape[2] == self.D
        assert q.device.type == "cuda" and k.device.type == "cuda" and v.device.type == "cuda"

        device = q.device
        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())

        # Output and LSE (float32 for numerical stability; cast output to bfloat16 at the end)
        output = torch.empty((total_q, self.G, self.D), dtype=torch.bfloat16, device=device)
        lse = torch.full((total_q,), -float("inf"), dtype=torch.float32, device=device)

        # Cast to float32 for compute
        q_f32 = q.contiguous().to(torch.float32)
        k_f32 = k.contiguous().to(torch.float32)
        v_f32 = v.contiguous().to(torch.float32)

        # Process each block b defined by qo_indptr
        for b in range(0, qo_indptr.numel() - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            M = max(q_end - q_start, 0)
            N = max(kv_end - kv_start, 0)

            if M == 0 or N == 0:
                continue

            # For each q position within the block, compute attention for all output heads
            for q_idx in range(0, M):
                q_idx_global = q_start + q_idx

                # Prepare Q vector for output head g=0
                q_vec_g0 = torch.empty((self.D,), dtype=torch.float32, device=device)
                compute_q_vec_kernel[(1,)](
                    q_f32, q_idx_global, 0, q_vec_g0, D=self.D
                )

                # K and V contiguous for this block: [N, GH, D]
                k_block = k_f32[kv_start:kv_end].contiguous()  # [N, GH, D]
                v_block = v_f32[kv_start:kv_end].contiguous()  # [N, GH, D]

                # Compute logits and LSE for gh=0
                out_logit = torch.empty((N,), dtype=torch.float32, device=device)
                lse_row = torch.empty((1,), dtype=torch.float32, device=device)
                attn_logits_lse_gh0_g0_kernel[(1,)](
                    q_vec_g0, k_block, v_block, out_logit, lse_row,
                    q_idx_global, N, D=self.D, GH=self.GH, BLOCK_N=N
                )
                lse[q_idx_global] = lse_row[0]  # row-wise LSE

                # Softmax and store output for each output head g
                for g_out in range(0, self.G):
                    softmax_and_store_out_kernel[(1,)](
                        out_logit, v_block, output,
                        q_idx_global, g_out, N, D=self.D, GH=self.GH
                    )

        # Cast output back to bfloat16 to match the original model
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
