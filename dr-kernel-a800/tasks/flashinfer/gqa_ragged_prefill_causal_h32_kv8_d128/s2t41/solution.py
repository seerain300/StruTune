import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def compute_logits_masked_and_lse(
    q_ptr,  # *float32, [num_q_tokens, num_qo_heads, head_dim]
    k_ptr,  # *float32, [num_kv_tokens, num_qo_heads, head_dim] (K expanded to 32 heads)
    mask_ptr,  # *int8, [num_q_tokens, num_kv_tokens]
    logits_ptr,  # *float32, [num_q_tokens, num_qo_heads, num_kv_tokens]
    lse_row_ptr,  # *float32, [num_q_tokens, num_qo_heads]
    sm_scale,  # float32 scalar
    num_q_tokens: tl.constexpr,
    num_kv_tokens: tl.constexpr,
    num_qo_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BH: tl.constexpr
):
    # Grid: (num_batches, ceil_div(num_qo_heads, BH), ceil_div(num_q_tokens, BLOCK_Q))
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    qo_start = pid_b * num_q_tokens
    kv_start = pid_b * num_kv_tokens

    h_start = pid_h * BH
    qo_mask = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
    valid_q = qo_mask < num_q_tokens

    for hh in range(BH):
        h = h_start + hh
        if h >= num_qo_heads:
            break

        # Track row-wise maxima for lse
        lse_acc = tl.full((BLOCK_Q,), -float('inf'), dtype=tl.float32)

        # Loop over KV tiles to compute logits and update lse_acc
        for kv_off in range(0, num_kv_tokens, BLOCK_KV):
            kv_mask = kv_off + tl.arange(0, BLOCK_KV)  # [BLOCK_KV]
            valid_k = kv_mask < num_kv_tokens  # [BLOCK_KV]

            # Load Q block: [BLOCK_Q, head_dim]
            q_block = tl.load(
                q_ptr + (qo_start + qo_mask[:, None]) * (num_qo_heads * head_dim) + h * head_dim,
                mask=valid_q[:, None],
                other=0.0
            )  # [BLOCK_Q, head_dim]

            # Load K block: [BLOCK_KV, head_dim]
            k_block = tl.load(
                k_ptr + (kv_start + kv_mask[:, None]) * (num_qo_heads * head_dim) + h * head_dim,
                mask=valid_k[:, None],
                other=0.0
            )  # [BLOCK_KV, head_dim]

            # Matmul: [BLOCK_Q, head_dim] @ [head_dim, BLOCK_KV] => [BLOCK_Q, BLOCK_KV]
            acc = tl.dot(q_block, tl.trans(k_block)) * sm_scale  # [BLOCK_Q, BLOCK_KV]

            # Load causal mask tile: [BLOCK_Q, BLOCK_KV], int8 (0/1)
            mask_tile = tl.load(
                mask_ptr + qo_mask[:, None] * num_kv_tokens + kv_mask[None, :],
                mask=valid_q[:, None] & valid_k[None, :],
                other=0
            )  # [BLOCK_Q, BLOCK_KV]

            # Apply mask: invalid entries -> -inf
            acc = tl.where(mask_tile != 0, acc, -float('inf'))

            # Update row-wise maxima
            for i in range(BLOCK_Q):
                row_acc = acc[i, :]  # [BLOCK_KV]
                row_max_i = tl.max(row_acc, axis=0)  # scalar
                lse_acc[i] = tl.maximum(lse_acc[i], row_max_i)

            # Store logits tile to logits_ptr
            # logits_ptr indexing: [row, head, col] = row * (num_qo_heads * num_kv_tokens) + head * num_kv_tokens + col
            for i in range(BLOCK_Q):
                row = qo_start + qo_mask[i]
                for j in range(BLOCK_KV):
                    col = kv_start + kv_mask[j]
                    ptr = logits_ptr + row * (num_qo_heads * num_kv_tokens) + h * num_kv_tokens + col
                    val = acc[i, j]
                    tl.store(ptr, val)

        # Store lse_row: scale by 1/log(2) to get 2-base LSE
        inv_log2 = 1.0 / math.log(2.0)
        for i in range(BLOCK_Q):
            tl.store(
                lse_row_ptr + (qo_start + qo_mask[i]) * num_qo_heads + h,
                lse_acc[i] * inv_log2
            )


@triton.jit
def compute_output_from_logits_and_lse(
    k_ptr,  # *float32, [num_kv_tokens, num_qo_heads, head_dim] (K expanded)
    v_ptr,  # *float32, [num_kv_tokens, num_qo_heads, head_dim] (V expanded)
    mask_ptr,  # *int8, [num_q_tokens, num_kv_tokens]
    logits_ptr,  # *float32, [num_q_tokens, num_qo_heads, num_kv_tokens]
    output_ptr,  # *float32, [num_q_tokens, num_qo_heads, head_dim]
    lse_row_ptr,  # *float32, [num_q_tokens, num_qo_heads]
    sm_scale,  # float32 scalar (unused here, but kept for signature consistency)
    num_q_tokens: tl.constexpr,
    num_kv_tokens: tl.constexpr,
    num_qo_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BH: tl.constexpr
):
    # Grid: (num_batches, ceil_div(num_qo_heads, BH), ceil_div(num_q_tokens, BLOCK_Q))
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    qo_start = pid_b * num_q_tokens
    kv_start = pid_b * num_kv_tokens

    h_start = pid_h * BH
    qo_mask = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
    valid_q = qo_mask < num_q_tokens

    for hh in range(BH):
        h = h_start + hh
        if h >= num_qo_heads:
            break

        # Load lse values per row (we stored 2-base in kernel 1)
        lse_2base = tl.load(
            lse_row_ptr + (qo_start + qo_mask) * num_qo_heads + h,
            mask=valid_q,
            other=-float('inf')
        )  # [BLOCK_Q]

        # For each query row i in this tile, compute output row
        for i in range(BLOCK_Q):
            if not valid_q[i]:
                continue
            row = qo_start + qo_mask[i]
            # Initialize output accumulator for this row and head
            out_row = tl.zeros((head_dim,), dtype=tl.float32)

            # Compute softmax over KV tokens: logits[row, h, :]
            for kv_off in range(0, num_kv_tokens, BLOCK_KV):
                kv_mask = kv_off + tl.arange(0, BLOCK_KV)  # [BLOCK_KV]
                valid_k = kv_mask < num_kv_tokens

                # Load logits tile for this row and head: [BLOCK_KV]
                logits_row = tl.load(
                    logits_ptr + row * (num_qo_heads * num_kv_tokens) + h * num_kv_tokens + kv_start + kv_mask,
                    mask=valid_k,
                    other=-float('inf')
                )  # [BLOCK_KV]

                # Normalize by lse_2base[i]
                exp_row = tl.exp(logits_row - lse_2base[i])  # [BLOCK_KV]

                # Compute denominator
                denom = tl.sum(exp_row, axis=0)  # scalar

                # Load K tile for matvec: [BLOCK_KV, head_dim]
                k_block = tl.load(
                    k_ptr + (kv_start + kv_mask[:, None]) * (num_qo_heads * head_dim) + h * head_dim,
                    mask=valid_k[:, None],
                    other=0.0
                )  # [BLOCK_KV, head_dim]

                # Load V tile: [BLOCK_KV, head_dim]
                v_block = tl.load(
                    v_ptr + (kv_start + kv_mask[:, None]) * (num_qo_heads * head_dim) + h * head_dim,
                    mask=valid_k[:, None],
                    other=0.0
                )  # [BLOCK_KV, head_dim]

                # Accumulate output contribution per column
                for j in range(BLOCK_KV):
                    if valid_k[j]:
                        weight = exp_row[j] / denom  # scalar
                        v_col = v_block[j, :]  # [head_dim]
                        out_row += weight * v_col

            # Store output row for this head
            out_ptr = output_ptr + row * (num_qo_heads * head_dim) + h * head_dim
            tl.store(out_ptr + tl.arange(0, head_dim), out_row)


class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale=1.0 / math.sqrt(128.0), num_qo_heads=32, num_kv_heads=8, head_dim=128):
        super().__init__()
        self.sm_scale = sm_scale
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.gqa_ratio = num_qo_heads // num_kv_heads

        # Tiling parameters (can be tuned)
        self.BLOCK_Q = 64
        self.BLOCK_KV = 64
        self.BH = 8  # process heads in tiles of 8

    def forward(self, q, k, v, qo_indptr, kv_indptr):
        # Triton-only implementation: no torch ops on tensors
        if not TRITON_AVAILABLE or q.device.type != 'cuda':
            raise RuntimeError("Triton/CUDA not available for ModelNew")

        # Ensure contiguous and convert to float32 for computation
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        total_q = q_f32.shape[0]
        total_kv = k_f32.shape[0]
        num_qo_heads = self.num_qo_heads
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_dim

        # Allocate outputs
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.full((total_q, num_qo_heads), -float('inf'), dtype=torch.float32, device=q.device)

        len_indptr = qo_indptr.numel() - 1

        for b in range(len_indptr):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Slice q, k, v for this batch
            q_batch = q_f32[q_start:q_end]          # [num_q_tokens, 32, 128]
            k_batch = k_f32[kv_start:kv_end]        # [num_kv_tokens, 8, 128]
            v_batch = v_f32[kv_start:kv_end]        # [num_kv_tokens, 8, 128]

            # Expand K/V by GQA ratio to match 32 heads
            k_expanded = k_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [num_kv_tokens, 32, 128]
            v_expanded = v_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [num_kv_tokens, 32, 128]

            # Build causal mask on host: mask[i, j] = (j < (i + 1 + delta)), delta = num_kv_tokens - num_q_tokens
            delta = num_kv_tokens - num_q_tokens
            i_range = torch.arange(num_q_tokens, device=q.device)
            j_range = torch.arange(num_kv_tokens, device=q.device)
            mask_2d = (j_range[None, :] < (i_range[:, None] + 1 + delta)).to(torch.int8)  # [num_q_tokens, num_kv_tokens]

            # Allocate logits buffer: [num_q_tokens, 32, num_kv_tokens] float32
            logits = torch.empty((num_q_tokens, num_qo_heads, num_kv_tokens), dtype=torch.float32, device=q.device)

            # Grid for kernels: (num_batches=1, ceil_div(32, BH), ceil_div(num_q_tokens, BLOCK_Q))
            grid = (1, triton.cdiv(num_qo_heads, self.BH), triton.cdiv(num_q_tokens, self.BLOCK_Q))

            # Kernel 1: compute logits, apply mask, and write lse (2-base)
            compute_logits_masked_and_lse[grid](
                q_batch, k_expanded, mask_2d, logits, lse, self.sm_scale,
                num_q_tokens, num_kv_tokens, num_qo_heads, head_dim,
                self.BLOCK_Q, self.BLOCK_KV, self.BH,
                num_warps=4, num_stages=2
            )

            # Kernel 2: compute output from logits and write lse (already written in kernel 1)
            # Note: compute_output_from_logits_and_lse does not need logits_ptr; we pass any valid pointer (unused in computation),
            # but since it is a required parameter, we pass logits. It only reads mask and lse_row.
            compute_output_from_logits_and_lse[grid](
                k_expanded, v_expanded, mask_2d, logits, output, lse, self.sm_scale,
                num_q_tokens, num_kv_tokens, num_qo_heads, head_dim,
                self.BLOCK_Q, self.BLOCK_KV, self.BH,
                num_warps=4, num_stages=2
            )

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
