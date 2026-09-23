import torch
import math


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    # Shapes and assertions
    total_q, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64

    # Prepare cached key-value tensors (squeeze the 1-sized dim and cast to fp32 for compute)
    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

    # Output buffers
    output = torch.zeros(
        (total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=q_nope.device
    )
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=q_nope.device)

    # Process each batch (qo_indptr encodes query ranges per batch)
    b = 0
    while b < qo_indptr.shape[0] - 1:
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        if q_start >= q_end:
            b += 1
            continue

        # Find corresponding token indices for this batch in kv_indices
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        if page_beg >= page_end:
            b += 1
            continue

        tok_idx = kv_indices[page_beg:page_end]  # [L]
        L = tok_idx.numel()
        if L == 0:
            b += 1
            continue

        # Gather cached keys for this batch
        Kc = Kc_all[tok_idx]  # [L, 512]
        Kp = Kp_all[tok_idx]  # [L, 64]

        # Loop over queries within this batch
        for i in range(q_start, q_end):
            # Current query vectors
            qn = q_nope[i].to(torch.float32)  # [16, 512]
            qp = q_pe[i].to(torch.float32)    # [16, 64]

            # Compute per-head logits: qn @ Kc.T + qp @ Kp.T
            logits_qn = qn @ Kc.transpose(0, 1)  # [16, L]
            logits_qp = qp @ Kp.transpose(0, 1)  # [16, L]
            logits = logits_qn + logits_qp
            logits_scaled = logits * sm_scale

            # Apply causal mask per row: j > query_abs_pos
            # query_abs_pos = L - (q_end - q_start) + i
            query_abs_pos = L - (q_end - q_start) + i
            if query_abs_pos < 0:
                mask = torch.ones((logits_scaled.shape[1],), dtype=torch.bool, device=logits_scaled.device)
            else:
                mask = torch.arange(L, device=logits_scaled.device) > query_abs_pos
            # Set invalid positions to -inf
            # Note: masked_fill_ expects a tensor, not a boolean mask in this context; use where
            logits_scaled = torch.where(mask.unsqueeze(0), logits_scaled, torch.tensor(float("-inf"), device=logits_scaled.device))

            # Compute lse = logsumexp(logits) / log(2)
            m = logits_scaled.max(dim=-1, keepdim=True).values
            sumexp = logits_scaled - m
            # Zero out invalid positions before exp
            sumexp = torch.where(mask.unsqueeze(0), sumexp, torch.tensor(0.0, device=logits_scaled.device))
            sumexp = sumexp.exp().sum(dim=-1, keepdim=True)
            lse[i] = (m + sumexp.log()) / math.log(2.0)

            # Softmax across L
            softmax = torch.softmax(logits_scaled, dim=-1)

            # Output = softmax @ Kc
            out = softmax @ Kc  # [16, 512]
            output[i] = out.to(torch.bfloat16)

        b += 1

    return output, lse


def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16)
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16)
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16)
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [34], dtype=torch.int32)
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7):
    _out = run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect the same inputs as 'run'
        return run(*args)


def run(*args):
    return ModelNew()(*args)
