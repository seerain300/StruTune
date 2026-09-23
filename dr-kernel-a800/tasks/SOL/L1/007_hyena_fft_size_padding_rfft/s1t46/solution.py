import torch

class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Pure torch implementation that mirrors the original behavior:
        - Input: x of shape (batch, channels, seqlen)
        - Output: (batch, channels, seqlen+1) real and imaginary parts separately (float32)
        """
        # Ensure we have exactly one input tensor
        # (The original Model.forward uses *args, but here we expect a single tensor)
        assert len(x.shape) == 3, "Input must be a 3D tensor of shape (batch, channels, seqlen)"
        batch, channels, seqlen = x.shape

        # Compute FFT size and ensure float32 for numerical stability (matches original)
        fft_size = 2 * seqlen
        x_f32 = x.to(torch.float32)

        # Compute rfft along the last dimension (per (batch, channel) slice)
        # Output shape: (batch, channels, seqlen + 1) complex
        x_freq = torch.fft.rfft(x_f32, n=fft_size, dim=-1)

        # Normalize by fft_size (2 * seqlen)
        x_freq = x_freq / fft_size

        # Extract real and imaginary parts
        x_freq_real = x_freq.real.contiguous()
        x_freq_imag = x_freq.imag.contiguous()

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
