//! Inverse complex FFT (icfft), which Burn does not provide natively.
//!
//! Copied from `sciml_rs::neural_operators::utils::fft` - the only spectral
//! helper `SpectralConv`'s forward pass needs beyond `burn::tensor::signal`.

use burn::tensor::{Tensor, signal};

/// Inverse complex FFT for a general (possibly non-Hermitian) spectrum, at
/// arbitrary rank `D`, via `conj(ifft(x)) = fft(conj(x)) / n`: negate the
/// imaginary part, run forward `cfft`, negate and scale the result back.
pub fn icfft_full_spectrum<const D: usize>(
    re: Tensor<D>,
    im: Tensor<D>,
    dim: usize,
    n: usize,
) -> (Tensor<D>, Tensor<D>) {
    // conj(x) = (re, -im)
    let conj_im = im.neg();

    let (fwd_re, fwd_im) = signal::cfft(re, conj_im, dim, Some(n));

    // conj(result) = (fwd_re, -fwd_im), scaled by 1/n
    let scale = 1.0 / n as f64;
    let out_re = fwd_re.mul_scalar(scale);
    let out_im = fwd_im.neg().mul_scalar(scale);

    (out_re, out_im)
}
