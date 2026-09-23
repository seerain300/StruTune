import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_kernel(
    encoder_ptr,       # *f32 [B, T, H]
    hidden_ptr,        # *f32 [B, I, H]
    out_ptr,           # *f32 [B, T+I, H]
    B: tl.int32,
    T: tl.int32,
    I: tl.int32,
    H: tl.int32,
    # Strides (in elements)
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    hidden_stride_b, hidden_stride_i, hidden_stride_h,
    out_stride_b, out_stride_t, out_stride_h,
    BLOCK_H: tl.constexpr,
):
    # program ids: batch and output token
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    # bounds check
    if (pid_b >= B) or (pid_t >= (T + I)):
        return

    # choose source: encoder if pid_t < T, else hidden at pid_t - T
    is_encoder = pid_t < T

    # iterate over H in tiles
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        # load source vector (scalar over h) with mask
        if is_encoder:
            ptr_in = encoder_ptr + pid_b * encoder_stride_b + pid_t * encoder_stride_t + offs_h * encoder_stride_h
            x = tl.load(ptr_in, mask=mask_h, other=0.0)
        else:
            ptr_in = hidden_ptr + pid_b * hidden_stride_b + (pid_t - T) * hidden_stride_i + offs_h * hidden_stride_h
            x = tl.load(ptr_in, mask=mask_h, other=0.0)

        # store to output
        ptr_out = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t + offs_h * out_stride_h
        tl.store(ptr_out, x, mask=mask_h)


@triton.jit
def matvec_kernel(
    in_ptr,            # *f32 [B, T+I, H]
    weight_ptr,        # *f32 [H, H]
    out_ptr,           # *f32 [B, T+I, H]
    B: tl.int32,
    T_I: tl.int32,     # T + I
    H: tl.int32,
    # Strides
    in_stride_b, in_stride_t, in_stride_h,
    weight_stride_w, weight_stride_k,
    out_stride_b, out_stride_t, out_stride_h,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # one program per (b, t)
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    if (pid_b >= B) or (pid_t >= T_I):
        return

    # Accumulator for this t over H
    acc = tl.zeros((H,), dtype=tl.float32)

    # Tile over H
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        # Accumulate acc[offs_h] += sum over K of in[b, t, k] * weight[offs_h, k]
        for k_off in range(0, H, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H

            # Load in[b, t, k] as vector
            ptr_in = in_ptr + pid_b * in_stride_b + pid_t * in_stride_t + offs_k * in_stride_h
            x = tl.load(ptr_in, mask=mask_k, other=0.0)  # [BLOCK_K]

            # Load weight block [BLOCK_H, BLOCK_K]: weight[offs_h, offs_k]
            ptr_w = weight_ptr + offs_h[:, None] * weight_stride_w + offs_k[None, :] * weight_stride_k
            w = tl.load(ptr_w, mask=mask_h[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_H, BLOCK_K]

            # Multiply and reduce over K
            # w: [BH, BK], x: [BK] -> [BH, BK] then sum over BK -> [BH]
            prod = w * x[None, :]
            acc_tile = tl.sum(prod, axis=1)

            # Add to acc
            acc = acc + acc_tile

        # Store acc for this t
        ptr_out = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t + offs_h * out_stride_h
        tl.store(ptr_out, acc, mask=mask_h)


@triton.jit
def split_encoder_kernel(
    src_ptr,           # *f32 [B, T+I, H]
    out_ptr,           # *f32 [B, T, H]
    B: tl.int32,
    T: tl.int32,
    I: tl.int32,
    H: tl.int32,
    # Strides
    src_stride_b, src_stride_t, src_stride_h,
    out_stride_b, out_stride_t, out_stride_h,
    BLOCK_H: tl.constexpr,
):
    # grid: (B, T)
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    if (pid_b >= B) or (pid_t >= T):
        return

    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        ptr_src = src_ptr + pid_b * src_stride_b + pid_t * src_stride_t + offs_h * src_stride_h
        x = tl.load(ptr_src, mask=mask_h, other=0.0)

        ptr_out = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t + offs_h * out_stride_h
        tl.store(ptr_out, x, mask=mask_h)


@triton.jit
def split_hidden_kernel(
    src_ptr,           # *f32 [B, T+I, H]
    out_ptr,           # *f32 [B, I, H]
    B: tl.int32,
    T: tl.int32,
    I: tl.int32,
    H: tl.int32,
    # Strides
    src_stride_b, src_stride_t, src_stride_h,
    out_stride_b, out_stride_t, out_stride_h,
    BLOCK_H: tl.constexpr,
):
    # grid: (B, I)
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    if (pid_b >= B) or (pid_i >= I):
        return

    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        # src index for hidden stream is t = pid_i + T
        ptr_src = src_ptr + pid_b * src_stride_b + (pid_i + T) * src_stride_t + offs_h * src_stride_h
        x = tl.load(ptr_src, mask=mask_h, other=0.0)

        ptr_out = out_ptr + pid_b * out_stride_b + pid_i * out_stride_t + offs_h * out_stride_h
        tl.store(ptr_out, x, mask=mask_h)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        1) Concatenate encoder_hidden_states and hidden_states along the sequence dimension (T + I).
        2) Apply linear projection via matvec kernel: out = cat @ process_weight.T.
        3) Split outputs back into encoder and hidden parts.
        Returns (processed_encoder_hidden_states, processed_hidden_states).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = encoder_hidden_states.shape[2]
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]."

        # Allocate concatenated tensor [B, T+I, H]
        out_concat = torch.empty((B, T + I, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch concatenation kernel: grid = (B, T+I)
        BLOCK_H = 128
        concat_seq_kernel[(B, T + I)](
            encoder_hidden_states, hidden_states, out_concat,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            out_concat.stride(0), out_concat.stride(1), out_concat.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=4,
        )

        # Allocate output of matvec [B, T+I, H]
        total = torch.empty((B, T + I, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch matvec kernel: grid = (B, T+I)
        matvec_kernel[(B, T + I)](
            out_concat, process_weight, total,
            B, T + I, H,
            out_concat.stride(0), out_concat.stride(1), out_concat.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            total.stride(0), total.stride(1), total.stride(2),
            BLOCK_H=BLOCK_H, BLOCK_K=64,
            num_warps=4,
        )

        # Allocate outputs
        processed_encoder = torch.empty((B, T, H), device=hidden_states.device, dtype=hidden_states.dtype)
        processed_hidden = torch.empty((B, I, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch split kernels
        split_encoder_kernel[(B, T)](
            total, processed_encoder,
            B, T, I, H,
            total.stride(0), total.stride(1), total.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=4,
        )

        split_hidden_kernel[(B, I)](
            total, processed_hidden,
            B, T, I, H,
            total.stride(0), total.stride(1), total.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=4,
        )

        return processed_encoder, processed_hidden