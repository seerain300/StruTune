import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_lse_and_output_kernel(
    q_ptr,          # *bfloat16, [B, H, D], contiguous
    token_ids_ptr,  # *int32,    [B, T_MAX], contiguous
    output_ptr,     # *bfloat16, [B, H, D], contiguous
    lse_ptr,        # *float32,  [B, H]
    sm_scale,       # float32 scalar
    B: tl.constexpr,       # batch size
    H: tl.constexpr,       # num query heads
    D: tl.constexpr,       # head dim
    T_MAX: tl.constexpr,   # maximum number of tokens per batch
    gqa_ratio: tl.constexpr,  # H // N (e.g., 4)
):
    # Grid: (B, H)
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Base pointer to q[b, h, :]
    q_base = q_ptr + b * (H * D) + h * D
    q_vec = tl.load(q_base).to(tl.float32)  # [D]

    # GQA mapping: kv head for query head h
    kvh = h // gqa_ratio  # 0..7 for h 0..31

    # Per-(b,h) accumulators: scalar max and sum, and vector output accumulator
    max_val = -float("inf")
    sum_exp = 0.0
    out_accum = tl.zeros((D,), dtype=tl.float32)

    # Loop over tokens (static, up to T_MAX)
    for t in range(T_MAX):
        tok_id = tl.load(token_ids_ptr + b * T_MAX + t).to(tl.int32)
        # If tok_id < 0 (padding), skip contribution
        if tok_id >= 0:
            # Compute pointer to k_row and v_row for this tok_id and kvh
            # We assume k_ptr, v_ptr are 1D vectors of length N*D, and kvh * D points to the start of that row.
            k_row_ptr = k_ptr + kvh * D
            v_row_ptr = v_ptr + kvh * D

            # Load k_vec and v_vec as 1D vectors
            k_vec = tl.load(k_row_ptr)  # but we only need elements 0:D, so we index with tl.arange
            idx = tl.arange(0, D)
            k_vec = tl.load(k_row_ptr + idx).to(tl.float32)  # [D]
            v_vec = tl.load(v_row_ptr + idx).to(tl.float32)  # [D]

            # Compute logits_scaled
            logits = tl.dot(q_vec, k_vec)  # scalar
            logits_scaled = logits * sm_scale

            # Update max and sum_exp
            max_val = tl.maximum(max_val, logits_scaled)
            exp_term = tl.exp(logits_scaled - max_val)
            sum_exp += exp_term
            out_accum += exp_term * v_vec

    # LSE in base-2
    inv_log2 = 1.0 / math.log(2.0)
    lse_b_h = max_val + tl.log(sum_exp) * inv_log2
    tl.store(lse_ptr + b * H + h, lse_b_h)

    # Store output: output[b, h, :] = out_accum / sum_exp (softmax normalization)
    out_out = (out_accum / sum_exp).to(tl.bfloat16)
    out_base = output_ptr + b * (H * D) + h * D
    tl.store(out_base, out_out)


# Helper to build token_ids_all: [B, T_MAX] from kv_indptr and kv_indices
def build_token_ids_all(kv_indptr, kv_indices, B, T_MAX, device):
    token_ids_list = []
    for b in range(B):
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        tokens = kv_indices[start:end].to(torch.int32) if end > start else torch.empty(0, dtype=torch.int32, device=device)
        pad = T_MAX - tokens.shape[0]
        if pad > 0:
            tokens = torch.nn.functional.pad(tokens, (0, 0, 0, pad), value=-1)
        else:
            tokens = tokens[:T_MAX]
        token_ids_list.append(tokens)
    return torch.stack(token_ids_list, dim=0)  # [B, T_MAX]


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure device and dtype
        device = q.device
        B, H, D = q.shape
        N = v_cache.shape[2]  # num_kv_heads, e.g., 8
        assert H == 32 and D == 128 and N == 8, "This Triton implementation expects H=32, D=128, N=8"
        assert kv_indptr.shape[0] == B + 1, "kv_indptr must have shape [B+1]"

        # Compute max tokens across batches to set T_MAX
        max_tokens = 0
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            max_tokens = max(max_tokens, max(0, end - start))
        # Choose T_MAX as max_tokens, cap at 256 for safety
        T_MAX = int(max_tokens) if max_tokens <= 256 else 256

        # Build token_ids_all [B, T_MAX]
        token_ids_all = build_token_ids_all(kv_indptr, kv_indices, B, T_MAX, device)

        # Prepare pointers: q is [B, H, D], k_ptr and v_ptr are flattened 1D pointers of length N*D (contiguous along D)
        q_contig = q.contiguous()  # [B, H, D]
        # For k_cache and v_cache, shape [P, 1, N, D] -> squeeze dim-1 gives [P, N, D], take P=0 (since we need one of P), but we need all P entries for tokens.
        # To avoid dynamic indexing in Triton, we prepack k_ptr and v_ptr per token as 1D vectors. Since in evaluation P is used through token_ids, we can construct per-token rows using token_ids_all and kvh mapping. Here, we assume k_cache and v_cache are provided per token via token_ids. We can create k_ptr and v_ptr by indexing k_cache and v_cache using token_ids_all per batch. However Triton kernel cannot read from k_cache inside; so we prepack them into two 1D tensors of length B*T_MAX*D.

        # Since we don't know num_tokens per batch in-kernel, we build k_ptr and v_ptr on host as follows:
        # For each b, allocate arrays for k_rows and v_rows of size T_MAX*D (we'll pad to T_MAX). Then fill them using token_ids_all[b, :] and kvh mapping.
        # This requires B separate host-side loops. But Triton only runs kernels; host-side loops are allowed (not computation). We implement this packing now.

        # Create k_ptr and v_ptr: [B, T_MAX, D] then


def run(*args):
    return ModelNew()(*args)
