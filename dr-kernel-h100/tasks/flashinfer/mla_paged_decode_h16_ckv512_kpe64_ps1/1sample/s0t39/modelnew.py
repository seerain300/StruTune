class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale=1.0):
        super().__init__()
        # Keep sm_scale as a default parameter to match the evaluator's optional 7th arg
        self.sm_scale = float(sm_scale)

    def forward(self, *args):
        # Accept 7 inputs: (q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)
        # Note: if the evaluator passes more than 7, we ignore extras by unpacking first 7.
        if len(args) < 7:
            raise RuntimeError("ModelNew.forward expects at least 7 inputs")
        q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale = args[:7]
        sm_scale = float(sm_scale)

        device = q_nope.device
        B, H, N = q_nope.shape
        _, _, Kp_dim = q_pe.shape

        # Prepare inputs: cast to fp32 and make contiguous
        q_nope_fp32 = q_nope.to(torch.float32).contiguous()        # [B, H, N]
        q_pe_fp32 = q_pe.to(torch.float32).contiguous()           # [B, H, Kp_dim]

        # Flatten caches to [num_pages, dim]
        num_pages = ckv_cache.shape[0]
        ckv_cache_fp32 = ckv_cache.to(torch.float32).contiguous().view(num_pages, N)  # [num_pages, N]
        kpe_cache_fp32 = kpe_cache.to(torch.float32).contiguous().view(num_pages, Kp_dim)  # [num_pages, Kp_dim]

        # Allocate outputs
        output_fp32 = torch.empty((B, H, N), dtype=torch.float32, device=device)  # we will fill via kernel
        lse = torch.empty((B, H), dtype=torch.float32, device=device)              # per-(b,h) lse

        # Process each batch element and head in Triton
        for b in range(B):
            # Compute token range per batch
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M_total = end - start
            if M_total <= 0:
                # No KV tokens for this batch element: output zeros and lse = -inf
                output_fp32[b] = torch.zeros((H, N), dtype=torch.float32, device=device)
                lse[b] = -float("inf")
                continue

            # Gather token indices for this batch element
            tok_idx = kv_indices[start:end].to(torch.int32).contiguous()  # [M_total]

            # Build Kc and Kp row pointers: flatten each Kc[m, :] and Kp[m, :] into 1D
            # Kc_row_ptrs: shape [M_total, N]
            Kc_row_ptrs = ckv_cache_fp32[tok_idx].contiguous().view(M_total * N)
            # Kp_row_ptrs: shape [M_total, Kp_dim]
            Kp_row_ptrs = kpe_cache_fp32[tok_idx].contiguous().view(M_total * Kp_dim)

            # Prepare qn row and qp row for this batch
            qn_row = q_nope_fp32[b].contiguous().view(N)            # [N]
            qp_row = q_pe_fp32[b].contiguous().view(Kp_dim)        # [Kp_dim]

            # Launch Triton kernel for this (b,)
            # Grid is 1 because we process one (b, h) per launch; we loop h below.
            for h in range(H):
                grid = (1,)
                compute_lse_and_output_kernel[grid](
                    qn_row, qp_row,
                    Kc_row_ptrs, Kp_row_ptrs,
                    M_total,
                    lse[b, h].view(1),                           # 1-element tensor to hold scalar lse
                    output_fp32[b, h],                          # output vector for this (b, h)
                    N=N, Kp_dim=Kp_dim, sm_scale=self.sm_scale
                )

        # Cast output to bfloat16 to match original function's output dtype
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse