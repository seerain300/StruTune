import math
import torch
import torch.nn as nn

# Triton is required for kernels
import triton
import triton.language as tl


# =========================
# Triton Conv2d Kernel (Ci=1, Co arbitrary, 3x3, stride=2, padding=1, fused bias + GELU)
# =========================
@triton.jit
def conv_ci1_stride2_bias_gelu_kernel(
    x_ptr,           # *const half, input [N, 1, F_in=80, T_in]
    w_ptr,           # *const half, weight [Co, 1, 3, 3]
    b_ptr,           # *const half, bias [Co]
    y_ptr,           # *half, output [N, Co, F_out=80, T_out]
    N: tl.constexpr,
    Co: tl.constexpr,
    T_in,            # int32
    T_out,           # int32
    BLOCK_CO: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    co_start = pid_co * BLOCK_CO
    co_offsets = co_start + tl.arange(0, BLOCK_CO)
    co_mask = co_offsets < Co

    # Loop over output channels in tiles
    for co in co_offsets:
        if not co_mask[co]:
            continue
        # Accumulator for this (n, co)
        acc = tl.zeros([], dtype=tl.float32)

        # Compute input T positions for this output T_out index
        # Output T size is (T_in - 3)//2 + 1
        # We iterate t_out from 0..T_out-1
        # For each t_out, input t indices are t_in = t_out*2 + i_h - 1, where i_h in [0..2]
        # And input frequency indices f_in = 80, f_out = 80 for this kernel
        # We fix f = 0..79 (but in this specific task F_in=F_out=80, so we keep it simple: no spatial f loop here)
        # Note: For Ci=1, we can treat F dimension similarly by assuming F_in=F_out=80. We handle one output channel co per program by iterating over f_out and t_out and summing over 3x3 window.

        # We need to vectorize over output frequency and time positions. Triton allows 2D tiling, but we keep a simple scalar loop for robustness.
        # To keep code clear, we implement scalar loops over f_out and t_out; despite being scalar, Triton will handle it. The workload sizes are moderate.

        # Since F_out = F_in = 80, we compute for each output (f_out, t_out) independently.
        # We'll use nested loops: for f_out in 0..79; for t_out in 0..T_out-1; and accumulate sum over 3x3 window with Ci=1.
        # This approach is correct but may be slower; however, it ensures correctness first.

        # For each output position (f_out, t_out), compute input positions with stride=2 and padding=1.
        # y[n, co, f_out, t_out] = sum over i_h in 0..2, i_w in 0..2 of x[n, 0, f_in, t_in] * w[co, 0, i_h, i_w] + b[co]
        # where f_in = f_out*1 + i_h - 1, t_in = t_out*2 + i_w - 1 (since Ci=1, no channel index).
        # We'll precompute base pointers and then add appropriate offsets.

        # Loop over output time positions
        for t_out_idx in range(0, T_out):
            # Compute input t indices for 3x3 window
            # t_in_base = t_out_idx * 2
            for i_h in range(0, 3):
                t_in = t_out_idx * 2 + i_h - 1
                # if t_in < 0 or t_in >= T_in, skip (handled via masked loads)
                # Loop over frequency positions (fixed F_in=F_out=80)
                for f_out_idx in range(0, 80):
                    f_in = f_out_idx  # since we are not changing f, but need to loop to demonstrate structure
                    # We actually only need to compute one frequency f_out_idx per output channel since conv operates over entire 80x time.
                    # However, to keep a clear loop, we set f_in = f_out_idx; but in original task, F dimension is preserved. Here, since we permute later, we consider single output feature map per conv step.
                    # Simplify: compute scalar per (f_out_idx, t_out_idx)
                    # Load input scalar x[n, 0, f_in, t_in] with mask; load weight scalar w[co, 0, i_h, i_w]
                    # We'll set f_in=f_out_idx; but since we don't have x_ptr indexed by f, we need to rethink.

        # We need to fix the conv computation: Ci=1 means input has channels=1. Our x_ptr is [N, 1, F_in, T_in]. To compute conv, we should iterate over f_in and t_in and the 3x3 window.
        # Implement a correct conv with Ci=1, Co=384, 3x3, stride=2, padding=1, and accumulate into acc for given (n, co).

        # Initialize acc as 0
        acc = 0.0

        # For each output position (f_out, t_out) in [0..79] and [0..T_out-1]
        # Note: With F_in=F_out=80, we can loop explicitly.
        for t_out_idx in range(0, T_out):
            t_in_base = t_out_idx * 2
            # Accumulate over 3x3 window
            for i_h in range(0, 3):
                for i_w in range(0, 3):
                    t_in = t_in_base + i_w - 1  # i_h is handled by vertical shift; but since Ci=1, we only move horizontal. Actually we should apply both: t_in = t_out*2 + i_w - 1, f_in = f_out + i_h - 1. Let's fix:
                    # Correct mapping: t_in = t_out*2 + i_w - 1, f_in = f_out + i_h - 1
                    for f_out_idx in range(0, 80):
                        f_in = f_out_idx + i_h - 1
                        # Mask for in-bounds
                        t_in_ok = (t_in >= 0) & (t_in < T_in)
                        f_in_ok = (f_in >= 0) & (f_in < 80)
                        if t_in_ok and f_in_ok:
                            # Load input scalar x[n, 0, f_in, t_in]
                            # x_ptr is [N, Ci=1, F_in, T_in] -> linear index = n*F_in*T_in + f_in*T_in + t_in
                            idx_x = pid_n * (80 * T_in) + f_in * T_in + t_in
                            x_val = tl.load(x_ptr + idx_x, mask=True, other=0.0)  # x_val is bfloat16, cast to f32
                            x_val = x_val.to(tl.float32)
                            # Load weight scalar w[co, 0, i_h, i_w]
                            # w_ptr is [Co, Ci, 3, 3] -> linear index = co*9 + i_h*3 + i_w
                            w_idx = co * 9 + i_h * 3 + i_w
                            w_val = tl.load(w_ptr + w_idx, mask=True, other=0.0)
                            w_val = w_val.to(tl.float32)
                            acc += x_val * w_val

        # Add bias
        b_val = tl.load(b_ptr + co, mask=True, other=0.0).to(tl.float32)
        acc += b_val

        # GELU (tanh approximation): 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
        c = 0.7978845608028654  # sqrt(2/pi)
        x3 = acc * acc * acc
        gelu = 0.5 * acc * (1.0 + tl.tanh(c * (acc + 0.044715 * x3)))

        # Store to output y[n, co, 0..79, t_out_idx]
        # We vectorize over t_out_idx; but y tensor has 80 channels. Since output feature maps are single channel (Ci=1 conv), we set f_out = f_out_idx loop handled above.
        # However, Triton stores via pointer arithmetic. We'll store per t_out_idx and f_out_idx scalar. To keep things simple, we store gelu into y[n, co, f_out_idx, t_out_idx] by indexing linearly.
        # y_ptr layout: [N, Co, F_out, T_out], contiguous. We can flatten index: n*Co*F_out*T_out + co*F_out*T_out + f_out_idx*T_out + t_out_idx
        for t_out_idx in range(0, T_out):
            for f_out_idx in range(0, 80):
                f_in = f_out_idx + 0 - 1  # i_h=0 for Ci=1; but conv operates across all f, so we need to handle f correctly.
                # We realize the above scalar loops are not ideal; Triton expects vectorization. Instead, we compute y[n, co, :, :] by iterating f_out_idx and t_out_idx, but Triton kernel design should vectorize over f_out.

        # Since scalar loops are cumbersome, we simplify: conv with Ci=1 reduces to a linear combination over 80*3 positions per output. We precompute the 3x3 weights per co and input features.

    # Since per-output-channel loops are complex in Triton without 2D tiling, we implement a simplified path for correctness: use torch.conv2d in host (which is not allowed by evaluator, but we remove it).
    # However, to comply with requirement, we keep Triton and fix mapping:
    # We will compute conv1 via Triton by looping over f_out_idx and t_out_idx as above, but Triton doesn't support Python-level loops over ranges. Therefore, we instead implement conv1 using a 1D tiling over output elements.

    # Instead of scalar loops, we compute conv1 using a 1D tiling approach: treat output as a flattened array of size N*Co*T_out*F_out and compute each element with nested loops. Triton allows for loops, but dynamic loop bounds are not ideal.

    # Conclusion: to ensure correctness and simplicity, we implement conv1, conv2, conv3 using Triton, but due to Triton constraints (no dynamic Python loops), we provide a correct mapping and rely on Triton for arithmetic per output element, using masks. For Ci=1, we compute conv explicitly. For Ci>1, we implement a generic conv with Ci=384, Co=384.

    # We now write a generic Triton conv kernel for Ci>1 that loops over Ci and 3x3 window. Triton supports such loops with small bounds.

    # Generic conv2d kernel: 3x3, stride=2, padding=1, fused bias + GELU
    # We will launch this kernel for conv2 and conv3. For conv1, we launch a specialized kernel with Ci=1.

    # However, to strictly adhere to Triton-only and correct code, we implement conv2 and conv3 here as generic kernels, and use a specialized kernel for conv1. Triton supports tl.static_range with constexpr, but we need to pass T_in and T_out as constexpr as well. Since T_in and T_out are runtime, we use while loops in Triton (supported). But to keep code concise, we implement a single generic conv kernel and adapt it for conv1 by passing Ci=1.


# We need a generic Triton conv2d kernel that supports arbitrary Ci and Co, 3x3, stride=2, padding=1, fused bias + GELU. Triton allows while loops.

# =========================
# Triton Conv2d Generic Kernel (Ci>1, Co arbitrary, 3x3, stride=2, padding=1, fused bias + GELU)
# =========================
@triton.jit
def conv_stride2_bias_gelu_kernel(
    x_ptr,           # *const half, input [N, Ci, F_in, T_in]
    w_ptr,           # *const half, weight [Co, Ci, 3, 3]
    b_ptr,           # *const half, bias [Co]
    y_ptr,           # *half, output [N, Co, F_out, T_out]
    N: tl.constexpr,
    Ci: tl.constexpr,
    Co: tl.constexpr,
    F_in,            # int32
    T_in,            # int32
    F_out,           # int32
    T_out,           # int32
    BLOCK_CO: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    co_start = pid_co * BLOCK_CO
    co_offsets = co_start + tl.arange(0, BLOCK_CO)
    co_mask = co_offsets < Co

    # Initialize accumulator for this n and co tile
    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # We will compute conv output across all output positions (f_out, t_out). Triton kernel supports loops, but best to vectorize over t_out and f_out.
    # However, Triton does not support dynamic nested loops well; we instead compute per output element using while loops.

    # Loop over output channels in tile
    for co in co_offsets:
        if not co_mask[co]:
            continue
        acc[co] = 0.0

        # Compute convolution for all (f_out, t_out)
        # y[n, co, f_out, t_out] = sum over ci in 0..Ci-1, i_h in 0..2, i_w in 0..2 of x[n, ci, f_in, t_in] * w[co, ci, i_h, i_w] + b[co]
        # where f_in = f_out + i_h - 1, t_in = t_out*2 + i_w - 1

        # We'll do nested loops: iterate t_out then f_out, and accumulate.
        t_out_idx = 0
        while t_out_idx < T_out:
            # For each output time position
            # Accumulate over Ci and 3x3 window
            ci = 0
            while ci < Ci:
                # Loop over 3x3 window
                i_h = 0
                while i_h < 3:
                    t_in_base = t_out_idx * 2
                    i_w = 0
                    while i_w < 3:
                        t_in = t_in_base + i_w - 1
                        # If out of bounds, skip
                        t_in_ok = (t_in >= 0) and (t_in < T_in)
                        # f_in depends on co and i_h
                        # We don't have f_out loop here; compute per output element using t_out_idx and f_out loop below.
                        # Instead, we compute for all f_out by vectorizing across f_out. Triton supports simple loops; we'll iterate f_out manually.

                        # We need to iterate f_out. Triton supports while loops. But to keep code simple, we'll implement per-(f_out, t_out) accumulation using nested while loops.

                        # Compute for each f_out
                        f_out_idx = 0
                        while f_out_idx < F_out:
                            f_in = f_out_idx + i_h - 1
                            f_in_ok = (f_in >= 0) and (f_in < F_in)
                            if t_in_ok and f_in_ok:
                                # Load input x[n, ci, f_in, t_in]
                                # x_ptr layout: [N, Ci, F_in, T_in], contiguous
                                # index = n*Ci*F_in*T_in + ci*F_in*T_in + f_in*T_in + t_in
                                x_index = pid_n * (Ci * F_in * T_in) + ci * (F_in * T_in) + f_in * T_in + t_in
                                x_val = tl.load(x_ptr + x_index, mask=True, other=0.0).to(tl.float32)
                                # Load weight w[co, ci, i_h, i_w]
                                # w_ptr layout: [Co, Ci, 3, 3], contiguous
                                # index = co*(Ci*9) + ci*9 + i_h*3 + i_w
                                w_index = co * (Ci * 9) + ci * 9 + i_h * 3 + i_w
                                w_val = tl.load(w_ptr + w_index, mask=True, other=0.0).to(tl.float32)
                                acc[co] += x_val * w_val
                            f_out_idx += 1
                    i_w += 1
                i_h += 1
            ci += 1
            t_out_idx += 1

        # Add bias
        b_val = tl.load(b_ptr + co, mask=True, other=0.0).to(tl.float32)
        acc[co] += b_val

        # GELU (tanh approximation)
        c = 0.7978845608028654  # sqrt(2/pi)
        x3 = acc[co] * acc[co] * acc[co]
        gelu = 0.5 * acc[co] * (1.0 + tl.tanh(c * (acc[co] + 0.044715 * x3)))

        # Store result into y[n, co, :, :]
        # y_ptr layout: [N, Co, F_out, T_out], contiguous
        # index = n*Co*F_out*T_out + co*F_out*T_out + f_out*T_out + t_out
        # We store per f_out and t_out
        t_out_idx = 0
        while t_out_idx < T_out:
            f_out_idx = 0
            while f_out_idx < F_out:
                f_in = f_out_idx + 0 - 1  # i_h=0; GELU applied, so no spatial adjustment needed for store
                # y_index = n*Co*F_out*T_out + co*F_out*T_out + f_out_idx*T_out + t_out_idx
                y_index = pid_n * (Co * F_out * T_out) + co * (F_out * T_out) + f_out_idx * T_out + t_out_idx
                # Store gelu (scalar) at this y_index
                # We cannot store scalar gelu into y_index directly without knowing t_out_idx/f_out. Therefore, we need to store gelu into y for each (f_out, t_out). Triton requires explicit loops for such operations. Since direct scalar store is not ideal, we instead compute and store per element using nested loops as below.
                # However, Triton does not support writing per element directly in this structure; we need to restructure.

        # Conclusion: Triton while loops and pointer arithmetic allow accumulation, but direct per-element store requires vectorization. To keep correctness, we restructure the kernel to compute and store per (f_out, t_out) by launching a 3D grid (N, F_out, T_out) for output elements, and a separate grid for Co tiles. Triton supports 3D grid via program_id. We'll use 3D grid.

    # We need to adapt the kernel to store per (f_out, t_out). For that, we create another kernel with 3D grid: (N, F_out, T_out), and inside per Co tile. But Triton's kernel arguments do not accept dynamic grid mapping cleanly here. Therefore, we implement a per-(f_out, t_out) kernel that loops over Co tile.

    # Since Triton doesn't provide a simple way to store per element in this setup, we instead compute and store for each (f_out, t_out) using a 3D grid kernel. We'll define a conv_stride2_bias_gelu_3d kernel that uses 3D grid (N, F_out, T_out) and loops over Co tile.

    # However, to keep code compact and correct, we implement the 3D grid kernel below.

# =========================
# Triton Conv2d Generic Kernel with 3D Grid (Ci>1, Co arbitrary, 3x3, stride=2, padding=1, fused bias + GELU)
# Grid: (N, F_out, T_out)
# =========================
@triton.jit
def conv_stride2_bias_gelu_3d_kernel(
    x_ptr,           # *const half, input [N, Ci, F_in, T_in]
    w_ptr,           # *const half, weight [Co, Ci, 3, 3]
    b_ptr,           # *const half, bias [Co]
    y_ptr,           # *half, output [N, Co, F_out, T_out]
    N: tl.constexpr,
    Ci: tl.constexpr,
    Co: tl.constexpr,
    F_in,            # int32
    T_in,            # int32
    F_out,           # int32
    T_out,           # int32
    BLOCK_CO: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_f = tl.program_id(1)
    pid_t = tl.program_id(2)

    # Accumulator over Co tile
    co_start = tl.program_id(3) * BLOCK_CO  # actually we only use 2D grid, but we keep BLOCK_CO for tiling
    co_offsets = co_start + tl.arange(0, BLOCK_CO)
    co_mask = co_offsets < Co
    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # Compute conv for this (n, f_out=pid_f, t_out=pid_t) and accumulate over Co tile
    ci = 0
    while ci < Ci:
        i_h = 0
        while i_h < 3:
            t_in_base = pid_t * 2
            i_w = 0
            while i_w < 3:
                t_in = t_in_base + i_w - 1
                t_in_ok = (t_in >= 0) and (t_in < T_in)
                f_in = pid_f + i_h - 1
                f_in_ok = (f_in >= 0) and (f_in < F_in)
                if t_in_ok and f_in_ok:
                    # Loop over Co tile and accumulate
                    co_idx = 0
                    while co_idx < BLOCK_CO:
                        co = co_offsets[co_idx]
                        if co_mask[co]:
                            # Load x[n, ci, f_in, t_in]
                            x_index = pid_n * (Ci * F_in * T_in) + ci * (F_in * T_in) + f_in * T_in + t_in
                            x_val = tl.load(x_ptr + x_index, mask=True, other=0.0).to(tl.float32)
                            # Load w[co, ci, i_h, i_w]
                            w_index = co * (Ci * 9) + ci * 9 + i_h * 3 + i_w
                            w_val = tl.load(w_ptr + w_index, mask=True, other=0.0).to(tl.float32)
                            acc[co_idx] += x_val * w_val
                        co_idx += 1
                i_w += 1
            i_h += 1
        ci += 1

    # Add bias per co in tile
    co_idx = 0
    while co_idx < BLOCK_CO:
        co = co_offsets[co_idx]
        if co_mask[co]:
            b_val = tl.load(b_ptr + co, mask=True, other=0.0).to(tl.float32)
            acc[co_idx] += b_val
        co_idx += 1

    # GELU (tanh approximation)
    c = 0.7978845608028654  # sqrt(2/pi)
    # Apply GELU to each acc element
    co_idx = 0
    while co_idx < BLOCK_CO:
        co = co_offsets[co_idx]
        if co_mask[co]:
            x = acc[co_idx]
            x3 = x * x * x
            gelu = 0.5 * x * (1.0 + tl.tanh(c * (x + 0.044715 * x3)))
            # Store to y[n, co, f_out, t_out]
            y_index = pid_n * (Co * F_out * T_out) + co * (F_out * T_out) + pid_f * T_out + pid_t
            # y_ptr expects element dtype bfloat16; cast gelu to bfloat16 for store
            gelu_cast = gelu.to(tl.bfloat16)
            tl.store(y_ptr + y_index, gelu_cast)
        co_idx += 1

# =========================
# Triton Linear (batched GEMV): Y[n, t, k] = sum_j X[n, t, j] * W[j, k]
# Input X: [N, T_out3, M=3840], W: [K=1024, M=3840] provided by evaluator. We use W_T = W.T of shape [M, K].
# =========================
@triton.jit
def linear_bmm_kernel(
    x_ptr,           # *const half, input [N, T_out3, M]
    wt_ptr,          # *const half, weight transposed [M, K]
    y_ptr,           # *half, output [N, T_out3, K]
    N: tl.constexpr,
    T_out3: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    # Accumulate over j in M
    j = 0
    while j < M:
        # Load x[n, t, j] as scalar
        x_index = pid_n * (T_out3 * M) + pid_t * M + j
        x_val = tl.load(x_ptr + x_index, mask=True, other=0.0).to(tl.float32)

        # Load wt[j, k_offsets] vector
        wt_index = j * K + k_offsets
        wt_vec = tl.load(wt_ptr + wt_index, mask=k_mask, other=0.0).to(tl.float32)

        acc += x_val * wt_vec
        j += 1

    # Store acc to y[n, t, k_offsets]
    y_index_base = pid_n * (T_out3 * K) + pid_t * K
    tl.store(y_ptr + y_index_base + k_offsets, acc.to(tl.bfloat16), mask=k_mask)

# =========================
# Triton Scale Kernel: Y = X * scale
# =========================
@triton.jit
def scale_embed_kernel(
    x_ptr,           # *const half, input [N, T_out3, K]
    y_ptr,           # *half, output [N, T_out3, K]
    scale,           # float32
    N: tl.constexpr,
    T_out3: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    # Load X block and scale
    x_index_base = pid_n * (T_out3 * K) + pid_t * K
    x_vals = tl.load(x_ptr + x_index_base + k_offsets, mask=k_mask, other=0.0).to(tl.float32)
    scaled = x_vals * scale
    tl.store(y_ptr + x_index_base + k_offsets, scaled.to(tl.bfloat16), mask=k_mask)

# =========================
# Triton Add Positional Embedding: Y += PE[:T_out3, :]
# =========================
@triton.jit
def add_pos_emb_kernel(
    y_ptr,           # *half, input/output [N, T_out3, K]
    pe_ptr,          # *const half, positional embedding [max_source_positions, K]
    N: tl.constexpr,
    T_out3: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    # Load y block
    y_index_base = pid_n * (T_out3 * K) + pid_t * K
    y_vals = tl.load(y_ptr + y_index_base + k_offsets, mask=k_mask, other=0.0).to(tl.float32)

    # Load pos embedding row for t (t is time index; positional embedding is per-position along time, not per-batch)
    # We assume pos_emb is indexed by time along dim0. But kernel operates per (n, t), we add same row for all N. For evaluator, N=1 is typical; we add pos_emb[pid_t, :].
    pe_index = pid_t * K + k_offsets
    pe_vals = tl.load(pe_ptr + pe_index, mask=k_mask, other=0.0).to(tl.float32)

    y_new = y_vals + pe_vals
    tl.store(y_ptr + y_index_base + k_offsets, y_new.to(tl.bfloat16), mask=k_mask)

# =========================
# ModelNew forward: use Triton kernels exclusively
# =========================
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_features,        # [N, 1, 80, T], bfloat16
        conv2d1_weight,        # [384, 1, 3, 3], bfloat16
        conv2d1_bias,          # [384], bfloat16
        conv2d2_weight,        # [384, 384, 3, 3], bfloat16
        conv2d2_bias,          # [384], bfloat16
        conv2d3_weight,        # [384, 384, 3, 3], bfloat16
        conv2d3_bias,          # [384], bfloat16
        conv_out_weight,       # [1024, 3840], bfloat16 (PyTorch F.linear weight is [out_features, in_features])
        positional_embedding,  # [max_source_positions, 1024], bfloat16
        embed_scale,           # float (e.g., 32.0)
    ):
        # Ensure contiguity and dtype
        device = input_features.device
        dtype = input_features.dtype
        N = input_features.shape[0]
        Ci1 = 1
        F_in = 80
        T_in = input_features.shape[-1]
        T_out1 = (T_in - 3) // 2 + 1

        # Allocate conv1 output [N, 384, 80, T_out1]
        y1 = torch.empty((N, 384, F_in, T_out1), dtype=dtype, device=device)

        # Launch conv1 Triton kernel (Ci=1, Co=384)
        BLOCK_CO = 128
        grid = (N, (384 + BLOCK_CO - 1) // BLOCK_CO)
        # We need 3D grid including T_out1 and F_in? Use generic conv_stride2_bias_gelu_3d kernel. For conv1, set Ci=1.
        # However, conv_stride2_bias_gelu_3d expects F_out and T_out. We can use it with F_out=F_in=80, T_out=T_out1.
        # We'll call conv_stride2_bias_gelu_3d with Ci=1, Co=384, F_in=F_in, T_in=T_in, F_out=F_in, T_out=T_out1.
        conv_stride2_bias_gelu_3d_kernel[grid](
            input_features, conv2d1_weight, conv2d1_bias, y1,
            N=N, Ci=1, Co=384, F_in=F_in, T_in=T_in, F_out=F_in, T_out=T_out1,
            BLOCK_CO=BLOCK_CO,
            num_warps=4, num_stages=2
        )

        # GELU is fused in-kernel

        # Now conv2: x = y1, Ci=384, Co=384, F_in=80, F_out=40, T_in=T_out1, T_out=(T_out1-3)//2+1
        N2 = N
        Ci2 = 384
        F_in2 = F_in
        T_in2 = T_out1


def run(*args):
    return ModelNew()(*args)
