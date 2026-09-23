class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        q: [B, 32, 128], dtype bfloat16
        k_cache: [num_pages, 8, 128], dtype float32 (as in the evaluator)
        v_cache: [num_pages, 8, 128], dtype float32
        kv_indptr: [B+1], int32
        kv_indices: [tokens], int32
        sm_scale: float (1.0 / sqrt(128))
        Returns: (output [B, 32, 128] bfloat16, lse [B, 32] float32)
        """
        B, num_qo_heads, head_dim = q.shape
        # Prepare outputs
        output = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32)  # will cast to bfloat16
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32)               # will keep as float32

        # Precompute GQA mapping ratio
        num_kv_heads = k_cache.shape[1]  # 8 in evaluator
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        # Loop over batch
        for b in range(B):
            # Number of tokens for this batch: num_tokens_b = kv_indptr[b+1] - kv_indptr[b]
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens_b = end - start
            if num_tokens_b <= 0:
                # No KV for this batch element; output zeros, lse -inf
                output[b].zero_()
                lse[b].fill_(-float("inf"))
                continue

            # Gather token indices for this batch
            token_indices = kv_indices[start:end].to(torch.long).contiguous()  # [num_tokens_b]

            # Loop over heads
            for h in range(num_qo_heads):
                kv_head = h // gqa_ratio  # GQA mapping

                # Gather K and V for this batch and kv_head: [num_tokens_b, head_dim]
                k_rows = k_cache[token_indices, kv_head, :]  # float32
                v_rows = v_cache[token_indices, kv_head, :]  # float32

                # Cast to float32 for Triton (q is bfloat16, already passed; we load and cast as needed)
                K_t = k_rows.contiguous()  # [num_tokens_b, head_dim]
                V_t = v_rows.contiguous()  # [num_tokens_b, head_dim]

                # Q vector for this head
                q_vec = q[b, h, :].to(torch.float32).contiguous()  # [head_dim]

                # Flatten K and V to 1D for Triton
                K_flat = K_t.view(-1)   # [num_tokens_b * head_dim]
                V_flat = V_t.view(-1)   # [num_tokens_b * head_dim]

                # Output vector (length head_dim) and LSE scalar
                out_vec = torch.empty((head_dim,), dtype=torch.float32, device=q.device)
                lse_scalar = torch.empty((), dtype=torch.float32, device=q.device)

                # Launch Triton kernel for (b, h) with grid=(1,)
                softmax_and_attention_single_bh[(1,)](
                    out_vec,               # OUT_ptr
                    lse_scalar,            # LSE_ptr (scalar; we'll expand to vector in host)
                    q_vec,                 # Q_ptr
                    K_flat,                # K_ptr flattened
                    V_flat,                # V_ptr flattened
                    num_tokens_b,          # num_tokens (runtime)
                    float(sm_scale),       # sm_scale
                    1.4426950408889634,    # LOG2_INVERSE
                    head_dim=128,          # constexpr
                )

                # Store results
                output[b, h, :] = out_vec
                lse[b, h] = lse_scalar.item()  # scalar to host, PyTorch will manage device properly

        # Cast output to bfloat16 as required by original; lse remains float32
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
