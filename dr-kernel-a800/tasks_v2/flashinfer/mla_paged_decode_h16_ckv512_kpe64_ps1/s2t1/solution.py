class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All inputs must be on CUDA device for Triton."

        B, H, D1 = q_nope.shape
        _, _, D2 = q_pe.shape
        assert H == 16, "num_qo_heads must be 16."
        assert D1 == 512, "head_dim_ckv must be 512."
        assert D2 == 64, "head_dim_kpe must be 64."
        assert ckv_cache.shape[1] == 1 and ckv_cache.shape[2] == D1, "ckv_cache expected shape [N, 1, 512]."
        assert kpe_cache.shape[1] == 1 and kpe_cache.shape[2] == D2, "kpe_cache expected shape [N, 1, 64]."

        device = q_nope.device

        len_indptr = kv_indptr.shape[0]
        assert len_indptr == B + 1, "kv_indptr length must be batch_size + 1."

        output = torch.empty((B, H, D1), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # One Triton program per batch element
        grid = (B,)

        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens = max(end - start, 0)

            # Prepare slices: q_nope_rows[b, :, :] -> [H, D1], q_pe_rows[b, :, :] -> [H, D2]
            qnb = q_nope[b].to(torch.float32).contiguous()  # [H, D1]
            qpb = q_pe[b].to(torch.float32).contiguous()    # [H, D2]

            # Prepare Kc_sub and Kp_sub: [L_tokens, D1] and [L_tokens, D2]
            if L_tokens > 0:
                idxs = kv_indices[start:start + L_tokens].to(torch.int32).contiguous()  # [L_tokens]
                Kc_sub = ckv_cache[idxs].squeeze(1).contiguous().to(torch.float32)     # [L_tokens, D1]
                Kp_sub = kpe_cache[idxs].squeeze(1).contiguous().to(torch.float32)     # [L_tokens, D2]
            else:
                Kc_sub = torch.empty((0, D1), dtype=torch.float32, device=device)
                Kp_sub = torch.empty((0, D2), dtype=torch.float32, device=device)

            _compute_heads_kernel[grid](
                qnb, qpb, Kc_sub, Kp_sub, output, lse,
                H=H,
                D1=D1,
                D2=D2,
                L_tokens=L_tokens,
                sm_scale=sm_scale,
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
