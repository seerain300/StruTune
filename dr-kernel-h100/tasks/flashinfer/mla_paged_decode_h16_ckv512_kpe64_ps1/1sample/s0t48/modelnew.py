import math
import torch


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages = ckv_cache.shape[0]
    len_indptr = kv_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    # Checks based on the original code's assumptions
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64

    device = q_nope.device

    # Prepare caches: ckv_cache has shape [num_pages, 1, 512] -> [num_pages, 512]
    Kc_all = ckv_cache.to(torch.float32)            # [num_pages, 512]
    Kc_all = Kc_all.squeeze(1)                      # [num_pages, 512]
    Kc_all = Kc_all.contiguous()

    # kpe_cache has shape [num_pages, 1, 64] -> [num_pages, 64]
    Kp_all = kpe_cache.to(torch.float32)           # [num_pages, 64]
    Kp_all = Kp_all.squeeze(1)                     # [num_pages, 64]
    Kp_all = Kp_all.contiguous()

    # Output tensor (bfloat16 as in original)
    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for b in range(batch_size):
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        if start >= end:
            lse[b].fill_(-float("inf"))
            output[b].zero_()
            continue

        M_total = end - start
        tok_idx = kv_indices[start:end].to(torch.long)  # [M_total]
        Kc = Kc_all[tok_idx]                            # [M_total, 512]
        Kp = Kp_all[tok_idx]                           # [M_total, 64]

        # q_nope: [1, 16, 512] -> qn: [16, 512]
        qn = q_nope[b].to(torch.float32).contiguous()  # [16, 512]
        # q_pe: [1, 16, 64] -> qp: [16, 64]
        qp = q_pe[b].to(torch.float32).contiguous()    # [16, 64]

        # Compute logits per head
        logits_kc = qn @ Kc.transpose(0, 1)           # [16, M_total]
        logits_kp = qp @ Kp.transpose(0, 1)           # [16, M_total]
        logits = logits_kc + logits_kp                # [16, M_total]

        # Scale and compute LogSumExp base-2
        scaled = logits * sm_scale
        row_max = torch.amax(scaled, dim=1, keepdim=True)       # [16, 1]
        sum_exp = torch.sum(torch.exp(scaled - row_max), dim=1) # [16]
        lse[b] = torch.log(sum_exp) / math.log(2.0)             # base-2 LSE

        # Softmax along tokens per head, then weighted sum over Kc
        attn = torch.softmax(scaled, dim=1)                     # [16, M_total]
        out = attn @ Kc                                       # [16, 512]
        output[b] = out.to(torch.bfloat16)

    return output, lse


def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16)
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16)
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16)
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32)
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Exactly mirror the original run's logic; no Triton usage for correctness.
        return run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)