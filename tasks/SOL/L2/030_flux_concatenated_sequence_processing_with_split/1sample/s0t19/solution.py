import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_sequences_kernel(
    encoder_hidden: tl.pointer,  # [B, T, H]
    hidden_states: tl.pointer,   # [B, I, H]
    out_cat: tl.pointer,         # [B, T+I, H]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
):
    b = tl.program_id(0)
    total = T + I
    for l in range(0, total):
        if l < T:
            row = b * T + l
            ptr = encoder_hidden + row * H + tl.arange(0, H)
            out_ptr = out_cat + b * (total * H) + l * H + tl.arange(0, H)
        else:
            row = b * I + (l - T)
            ptr = hidden_states + row * H + tl.arange(0, H)
            out_ptr = out_cat + b * (total * H) + l * H + tl.arange(0, H)
        # Copy H elements
        vals = tl.load(ptr)  # [H]
        tl.store(out_ptr, vals)


@triton.jit
def _batched_matmul_per_row_kernel(
    A: tl.pointer,  # [M, H], M = B*(T+I)
    W: tl.pointer,  # [H, H]
    C: tl.pointer,  # [M, H], output
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr, M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0)
    row = m
    acc = tl.zeros([H], dtype=tl.float32)
    # Iterate over K in tiles
    for k0 in range(0, H, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)
        # Load A[m, k0:k0+BLOCK_K]
        a = tl.load(A + row * H + k0 + tl.arange(0, BLOCK_K))  # [BLOCK_K]
        # Load W[k0:k0+BLOCK_K, :] as [BLOCK_K, H]
        w = tl.load(W + k0 + tl.arange(0, BLOCK_K)[:, None] * H + tl.arange(0, H)[None, :])  # [BLOCK_K, H]
        acc += tl.dot(a[:, None], w)[0]  # [H]
    # Store result
    tl.store(C + row * H + tl.arange(0, H), acc)


@triton.jit
def _split_into_encoder_hidden_kernel(
    C: tl.pointer,               # [M, H], M = B*(T+I)
    processed_encoder: tl.pointer,  # [B, T, H]
    processed_hidden: tl.pointer,   # [B, I, H]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
):
    b = tl.program_id(0)
    total = T + I
    # Copy first T rows to encoder
    for l in range(0, T):
        row = b * total + l
        src = C + row * H + tl.arange(0, H)
        dst = processed_encoder + b * T * H + l * H + tl.arange(0, H)
        vals = tl.load(src)
        tl.store(dst, vals)
    # Copy remaining I rows to hidden
    for l in range(0, I):
        row = b * total + T + l
        src = C + row * H + tl.arange(0, H)
        dst = processed_hidden + b * I * H + l * H + tl.arange(0, H)
        vals = tl.load(src)
        tl.store(dst, vals)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        # Ensure device and dtype compatibility; Triton requires CUDA tensors
        device = hidden_states.device
        dtype = torch.float32  # use fp32 for robustness
        B, T, H = encoder_hidden_states.shape
        I = hidden_states.shape[1]
        M = B * (T + I)

        # 1) Concatenate along sequence dimension in Triton
        out_cat = torch.empty((B, T + I, H), dtype=dtype, device=device)
        _concatenate_sequences_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B=B, T=T, I=I, H=H,
            num_warps=1, num_stages=1
        )

        # 2) Batched GEMM in Triton: C = out_cat @ process_weight.T
        # out_cat: [M, H], process_weight: [H, H] -> C: [M, H]
        A = out_cat
        W = process_weight  # [H, H]
        C = torch.empty((M, H), dtype=dtype, device=device)
        _batched_matmul_per_row_kernel[(M,)](
            A, W, C,
            B=B, T=T, I=I, H=H, M=M,
            BLOCK_K=128,
            num_warps=4, num_stages=2
        )

        # 3) Split into [B, T, H] and [B, I, H] in Triton
        processed_encoder = torch.empty((B, T, H), dtype=dtype, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=dtype, device=device)
        _split_into_encoder_hidden_kernel[(B,)](
            C, processed_encoder, processed_hidden,
            B=B, T=T, I=I, H=H,
            num_warps=1, num_stages=1
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
