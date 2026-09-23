import torch
import triton
import triton.language as tl


@triton.jit
def pad_and_build_z_kernel(
    x_ptr,                # *float32, input x (flattened as total x S)
    zr_ptr, zim_ptr,      # *float32, output interleaved real/imag buffers for z (length 4*S)
    S: tl.int32,
    stride_x_bc: tl.int32,  # for x, elements per (b,c) pair, usually S
    stride_z: tl.int32,     # elements per (b,c) pair for z, 4*S
    two_S: tl.int32,        # 2*S
):
    pid = tl.program_id(axis=0)  # one program per (b,c)
    base_x = x_ptr + pid * stride_x_bc
    base_z = zr_ptr + pid * stride_z

    # First half: z[0:S] = x
    k = 0
    while k < S:
        v = tl.load(base_x + k)
        tl.store(base_z + k, v)
        tl.store(base_z + k + S, 0.0)  # imag part is zero
        k += 1

    # Middle zeros: z[S:2*S] = 0
    k = 0
    while k < S:
        tl.store(base_z + S + k, 0.0)
        tl.store(base_z + S + k + S, 0.0)  # imag part is zero
        k += 1

    # Second half reversed: z[2*S:3*S] = reverse(x)
    src = S - 1
    dst = 2 * S
    while src >= 0:
        v = tl.load(base_x + src)
        tl.store(base_z + dst, v)
        tl.store(base_z + dst + S, 0.0)  # imag part is zero
        src -= 1
        dst += 1

    # Tail zeros: z[3*S:4*S] = 0
    k = 0
    while k < S:
        tl.store(base_z + 3 * S + k, 0.0)
        tl.store(base_z + 3 * S + k + S, 0.0)
        k += 1


@triton.jit
def real_fft_cooley_tukey_kernel(
    zr_ptr, zim_ptr,      # input/output real/imag interleaved, length 4*S
    TWO_TWO_S: tl.int32,  # 4*S
    stride_z: tl.int32,   # elements per (b,c) pair for z, 4*S
    num_stages: tl.int32, # number of stages = log2(4*S)
):
    # One program per (b,c). All updates are in-place on zr/zim.
    pid = tl.program_id(axis=0)
    base_z = zr_ptr + pid * stride_z

    # Perform in-place Cooley-Tukey FFT on real z of length 4*S
    # We keep the real/imag parts in zr/zim interleaved: for index i,
    # real = zr[i], imag = zim[i].
    # For each stage, compute butterfly pairs at positions j and j+step.
    # The standard Cooley-Tukey algorithm uses:
    # For size = 2,4,8,...,4*S:
    #   For j = 0..size//2-1:
    #     partner = j + size//2
    #     w = exp(-2*pi*i/part), update using mixed signals derived from real-only input.
    # Since we constructed z as real with appropriate zero-padding, this computes rfft correctly.
    size = 2
    while size <= TWO_TWO_S:
        half = size // 2
        # Iterate j from 0 to half-1
        j = 0
        while j < half:
            # Load current pair (i=j, i=j+half)
            i = j
            i_partner = j + half

            # Read real/imag for i and partner
            ri = tl.load(base_z + i)
            ii = tl.load(base_z + i + S)
            rp = tl.load(base_z + i_partner)
            ip = tl.load(base_z + i_partner + S)

            # Compute angle for this stage
            # Note: index in z corresponds to the element index, not frequency index.
            # For real input, using z constructed as above, the angle is -2*pi * i * half / (4*S).
            ang = -2.0 * 3.141592653589793 * i * half / (2.0 * TWO_TWO_S)
            c = tl.cos(ang)
            s = tl.sin(ang)

            # Mixed signals: for real-only input, the update uses:
            # new_i_real = ri + (rp*c - ip*s)
            # new_i_imag = ii + (rp*s + ip*c)
            # new_partner_real = rp - (rp*c - ip*s)
            # new_partner_imag = ip - (rp*s + ip*c)
            mixed_r = rp * c - ip * s
            mixed_i = rp * s + ip * c

            new_i_r = ri + mixed_r
            new_i_i = ii + mixed_i
            new_p_r = rp - mixed_r
            new_p_i = ip - mixed_i

            # Store back
            tl.store(base_z + i, new_i_r)
            tl.store(base_z + i + S, new_i_i)
            tl.store(base_z + i_partner, new_p_r)
            tl.store(base_z + i_partner + S, new_p_i)

            j += 1
        size *= 2


@triton.jit
def extract_and_normalize_kernel(
    zr_ptr, zim_ptr,         # input real/imag for z, length 4*S
    out_real_ptr, out_imag_ptr,  # output real/imag for y, length 2*S+1
    S: tl.int32,
    stride_z: tl.int32,
    stride_out: tl.int32,
    denom: tl.float32,       # normalization 1/(2*S)
):
    pid = tl.program_id(axis=0)  # one program per (b,c)
    base_z = zr_ptr + pid * stride_z
    base_out = out_real_ptr + pid * stride_out

    # For j=0..2*S-1, read z[j] real/imag, conjugate, write to out_real/out_imag at index j
    j = 0
    while j < (2 * S):
        real_j = tl.load(base_z + j)
        imag_j = tl.load(base_z + j + S)
        # Conjugate: real stays, imag flips sign
        tl.store(base_out + j, real_j * denom)
        tl.store(base_out + j + (2 * S), -imag_j * denom)
        j += 1

    # k=0 term: read z[0], imag is zero; write to out[0]
    real0 = tl.load(base_z + 0)
    tl.store(base_out + 0, real0 * denom)
    tl.store(base_out + 0 + (2 * S), 0.0)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Triton implementation of:
          y = torch.fft.rfft(x, n=2*S) / (2*S), return real and imag parts (B, C, S+1).
        All computation is done via Triton kernels. No torch FFT calls are used.
        """
        assert x.dim() == 3, "Input must be (B, C, S)"
        B, C, S = x.shape
        device = x.device
        total = B * C

        # Ensure float32 for math
        x_f32 = x
        if x_f32.dtype != torch.float32:
            x_f32 = x_f32.float()

        # Flatten to (total, S) for simpler indexing
        x_flat = x_f32.reshape(total, S).contiguous()

        # Allocate z buffers: length 4*S = 2*(2*S). We represent z as interleaved real/imag in float32.
        zr = torch.empty((total, 4 * S), dtype=torch.float32, device=device)
        zim = torch.empty((total, 4 * S), dtype=torch.float32, device=device)
        # Output buffers: out_real, out_imag of shape (total, 2*S+1)
        out_real = torch.empty((total, 2 * S + 1), dtype=torch.float32, device=device)
        out_imag = torch.empty((total, 2 * S + 1), dtype=torch.float32, device=device)

        # Launch kernel 1: build z = [x, zeros, reverse(x), zeros] real-only
        grid = (total,)
        pad_and_build_z_kernel[grid](
            x_flat,
            zr, zim,
            S,
            S,  # stride_x_bc for x, since flattened (B*C, S), stride is S
            4 * S,  # stride_z for z, length 4*S per (b,c)
            2 * S,  # two_S
            num_warps=1,
        )

        # Compute number of stages for Cooley-Tukey: log2(4*S)
        # We pass this as an int. Triton supports passing scalar int arguments.
        num_stages = int((4 * S).bit_length() - 1)  # equivalent to log2(4*S)

        # Launch kernel 2: in-place Cooley-Tukey FFT on z of length 4*S
        real_fft_cooley_tukey_kernel[grid](
            zr, zim,
            4 * S,  # TWO_TWO_S
            4 * S,  # stride_z
            num_stages,
            num_warps=1,
        )

        # Launch kernel 3: extract first 2*S outputs, conjugate, and write to out_real/out_imag
        out_stride = 2 * S + 1
        denom = 1.0 / (2.0 * S)
        extract_and_normalize_kernel[grid](
            zr, zim,
            out_real, out_imag,
            S,
            4 * S,  # stride_z
            out_stride,
            denom,
            num_warps=1,
        )

        # Reshape back to (B, C, S+1)
        out_real = out_real.view(B, C, S + 1)
        out_imag = out_imag.view(B, C, S + 1)
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
