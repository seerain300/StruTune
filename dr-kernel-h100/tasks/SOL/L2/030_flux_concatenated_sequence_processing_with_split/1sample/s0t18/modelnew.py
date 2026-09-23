import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_sequences_kernel(
    encoder_hidden: tl.pointer,  # [B, T, H]
    hidden_states: tl.pointer,   # [B, I, H]
    out_cat: tl.pointer,         # [B, T + I, H]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr
):
    b = tl.program_id(axis=0)
    # guard: in case grid > B (not needed if we set grid=B)
    if b >= B:
        return

    total = T + I
    # We use a simple loop over total sequence length. H is the last-dim size.
    for l in range(0, total):
        if l < T:
            # load from encoder_hidden[b, l, :]
            row_ptr = encoder_hidden + b * T * H + l * H
            out_ptr = out_cat + b * (T + I) * H + l * H
        else:
            # load from hidden_states[b, l - T, :]
            row_ptr = hidden_states + b * I * H + (l - T) * H
            out_ptr = out_cat + b * (T + I) * H + l * H
        # Copy H elements
        for h in range(0, H):
            val = tl.load(row_ptr + h)
            tl.store(out_ptr + h, val)


@triton.jit
def _split_into_encoder_hidden_kernel(
    C: tl.pointer,               # [M, H], M = B*(T+I)
    processed_encoder: tl.pointer,  # [B, T, H]
    processed_hidden: tl.pointer,   # [B, I, H]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr
):
    b = tl.program_id(axis=0)
    if b >= B:
        return
    total = T + I
    # For each l in [0, T), copy C[b*(T+I) + l, :] to processed_encoder[b, l, :]
    for l in range(0, T):
        row_idx = b * total + l
        src_ptr = C + row_idx * H
        dst_ptr = processed_encoder + b * T * H + l * H
        for h in range(0, H):
            val = tl.load(src_ptr + h)
            tl.store(dst_ptr + h, val)
    # For each l in [T, T+I), copy to processed_hidden[b, l - T, :]
    for l in range(T, total):
        row_idx = b * total + l
        src_ptr = C + row_idx * H
        dst_idx = l - T
        dst_ptr = processed_hidden + b * I * H + dst_idx * H
        for h in range(0, H):
            val = tl.load(src_ptr + h)
            tl.store(dst_ptr + h, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, H]
        encoder_hidden_states: [B, T, H]
        process_weight: [H, H]
        returns (processed_encoder: [B, T, H], processed_hidden: [B, I, H])
        """
        B = hidden_states.shape[0]
        I = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "hidden_dim must match"
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"

        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1) Concatenate along sequence dimension using Triton: out_cat [B, T + I, H]
        out_cat = torch.empty((B, T + I, H), dtype=dtype, device=device)
        _concatenate_sequences_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B=B, T=T, I=I, H=H,
            num_warps=1, num_stages=1
        )

        # 2) GEMM with torch for robustness: C = out_cat @ process_weight.T
        #    out_cat is [B*(T+I), H]; process_weight is [H, H]
        M = B * (T + I)
        C = torch.matmul(out_cat, process_weight.t())

        # 3) Split into [B, T, H] and [B, I, H] using Triton
        processed_encoder = torch.empty((B, T, H), dtype=dtype, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=dtype, device=device)
        _split_into_encoder_hidden_kernel[(B,)](
            C, processed_encoder, processed_hidden,
            B=B, T=T, I=I, H=H,
            num_warps=1, num_stages=1
        )

        return processed_encoder, processed_hidden