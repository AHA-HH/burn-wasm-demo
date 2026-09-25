//! Spectral convolution, generic over spatial dimensionality.
//!
//! Copied from `sciml_rs::neural_operators::layers::spectral_convolution` for
//! inference only - the core operation of a Fourier Neural Operator: transform
//! to the frequency domain, apply a learned linear map to the channel axis at
//! each retained frequency, and transform back. Because the map is applied
//! per-frequency rather than per-grid-point, the layer is a global convolution
//! and is independent of the input discretisation - the same weights apply at
//! any resolution, which is what makes the resolution slider work.
//!
//! # Rank
//!
//! `R` is the total tensor rank: `R = D + 2` for `D` spatial axes, plus the
//! batch and channel axes. Burgers is 1D, so this crate only ever instantiates
//! `SpectralConv<3>`.

use burn::{
    Tensor,
    module::{Module, Param},
    tensor::signal,
    tensor::{Device, Distribution},
};

use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use crate::fft::icfft_full_spectrum;

/// Learned spectral convolution over `R - 2` spatial axes.
#[derive(Module, Debug)]
pub struct SpectralConv<const R: usize> {
    /// One weight per mode corner; `len() == 2^(D-1)`.
    /// Each is `[in_channels, out_channels, modes[0], .., modes[D-1]]`.
    weights_re: Vec<Param<Tensor<R>>>,
    weights_im: Vec<Param<Tensor<R>>>,
    /// Retained frequency count per spatial axis; `len() == R - 2`.
    modes: Vec<usize>,
    flat_modes: usize,
}

impl<const R: usize> SpectralConv<R> {
    /// # Panics
    ///
    /// If `modes.len() != R - 2`.
    pub fn new(device: &Device, in_channels: usize, out_channels: usize, modes: &[usize]) -> Self {
        assert_eq!(
            modes.len(),
            R - 2,
            "modes must have one entry per spatial axis (R - 2)"
        );
        let modes = modes.to_vec();
        let flat_modes = modes.iter().product();
        let num_corners = 1usize << (modes.len() - 1);

        let mut shape = vec![in_channels, out_channels];
        shape.extend_from_slice(&modes);
        let shape: [usize; R] = shape.try_into().unwrap();

        // Scaling follows the reference implementation.
        let scale = 1.0 / (in_channels * out_channels) as f64;
        let mut weights_re = Vec::with_capacity(num_corners);
        let mut weights_im = Vec::with_capacity(num_corners);
        for _ in 0..num_corners {
            weights_re.push(Param::from_tensor(Tensor::<R>::random(
                shape,
                Distribution::Uniform(0.0, scale),
                device,
            )));
            weights_im.push(Param::from_tensor(Tensor::<R>::random(
                shape,
                Distribution::Uniform(0.0, scale),
                device,
            )));
        }

        Self {
            weights_re,
            weights_im,
            modes,
            flat_modes,
        }
    }

    /// Spatial-axis slices selecting one mode corner.
    ///
    /// Bit `j` of `mask` picks the low (`0`) or high (`1`) frequency block on
    /// axis `j`. The last axis is always low: the real FFT leaves it with only
    /// non-negative frequencies, so it has no mirror to select. This is why
    /// `mask` spans `2^(D-1)` values rather than `2^D`.
    fn corner_ranges(mask: usize, modes: &[usize], full_dims: &[usize]) -> Vec<Range<usize>> {
        let d = modes.len();
        (0..d)
            .map(|axis| {
                if axis == d - 1 || (mask >> axis) & 1 == 0 {
                    0..modes[axis]
                } else {
                    full_dims[axis] - modes[axis]..full_dims[axis]
                }
            })
            .collect()
    }

    /// Complex channel mixing at every retained frequency.
    ///
    /// `(a + bi)(c + di) = (ac - bd) + (ad + bc)i`, batched over modes.
    ///
    /// The mode axes are flattened to a single axis `M` so the contraction is a
    /// plain batched matmul `[M, B, I] @ [M, I, O]`. This keeps every
    /// intermediate at literal rank 3 - the broadcast-and-sum alternative would
    /// need rank `R + 1`, which stable Rust cannot express as a type.
    fn complex_multiplication(
        x_re: Tensor<R>,
        x_im: Tensor<R>, // [B, I, modes..]
        w_re: Tensor<R>,
        w_im: Tensor<R>, // [I, O, modes..]
        modes: &[usize],
        flat: usize,
    ) -> (Tensor<R>, Tensor<R>) {
        let (b, i) = (x_re.dims()[0], x_re.dims()[1]);
        let o = w_re.dims()[1];

        let x_re = x_re.reshape([b, i, flat]).permute([2, 0, 1]);
        let x_im = x_im.reshape([b, i, flat]).permute([2, 0, 1]);
        let w_re = w_re.reshape([i, o, flat]).permute([2, 0, 1]);
        let w_im = w_im.reshape([i, o, flat]).permute([2, 0, 1]);

        let ac = x_re.clone().matmul(w_re.clone());
        let bd = x_im.clone().matmul(w_im.clone());

        let ab_cd = (x_re + x_im).matmul(w_re + w_im);

        let out_re = ac.clone() - bd.clone();
        let out_im = ab_cd - ac - bd;

        let mut out_shape = vec![b, o];
        out_shape.extend_from_slice(modes);
        let out_shape: [usize; R] = out_shape.try_into().unwrap();

        (
            out_re.permute([1, 2, 0]).reshape(out_shape),
            out_im.permute([1, 2, 0]).reshape(out_shape),
        )
    }

    /// Forward N-d transform: real FFT on the last spatial axis, complex FFT on
    /// the rest. Halves the storage and the corner count versus a full complex
    /// transform, since a real signal's spectrum is Hermitian.
    fn fft_ctensor(x: Tensor<R>) -> (Tensor<R>, Tensor<R>) {
        let dims = x.dims(); // read before x is consumed
        let last = R - 1;
        let (mut re, mut im) = signal::rfft(x, last, Some(dims[last]));
        for axis in (2..last).rev() {
            let (r, i) = signal::cfft(re, im, axis, Some(dims[axis]));
            re = r;
            im = i;
        }
        (re, im)
    }

    /// Inverse of [`fft_ctensor`], applying the axes in the reverse order.
    fn ifft_ctensor(mut re: Tensor<R>, mut im: Tensor<R>, orig_dims: &[usize]) -> Tensor<R> {
        let last = R - 1;
        for (axis, &dim) in orig_dims.iter().enumerate().take(last).skip(2) {
            let (r, i) = icfft_full_spectrum(re, im, axis, dim);
            re = r;
            im = i;
        }
        signal::irfft(re, im, last, Some(orig_dims[last]))
    }

    /// `[B, in_channels, spatial..] -> [B, out_channels, spatial..]`.
    ///
    /// Spatial extents are preserved; the channel count changes.
    pub fn forward(&self, x: Tensor<R>) -> Tensor<R> {
        // Captured before the transform: irfft needs the original extent to
        // recover the correct length from the truncated half-spectrum.
        let orig_dims = x.dims();
        let (batch, in_ch) = (orig_dims[0], orig_dims[1]);
        let out_ch = self.weights_re[0].val().dims()[1];
        let num_corners = self.weights_re.len();

        let (x_ft_re, x_ft_im) = Self::fft_ctensor(x);
        // Post-transform extents - the last axis is now n/2 + 1.
        let spec_dims: Vec<usize> = x_ft_re.dims()[2..].to_vec();

        let mut out_shape = vec![batch, out_ch];
        out_shape.extend_from_slice(&spec_dims);
        let out_shape: [usize; R] = out_shape.try_into().unwrap();

        // Frequencies outside the retained modes stay zero - this is the
        // low-pass truncation that makes the operator resolution-independent.
        let mut out_ft_re = Tensor::<R>::zeros(out_shape, &x_ft_re.device());
        let mut out_ft_im = Tensor::<R>::zeros(out_shape, &x_ft_im.device());

        for corner in 0..num_corners {
            let ranges = Self::corner_ranges(corner, &self.modes, &spec_dims);

            let mut in_ranges = vec![0..batch, 0..in_ch];
            in_ranges.extend(ranges.clone());
            let in_ranges: [Range<usize>; R] = in_ranges.try_into().unwrap();

            let x_re = x_ft_re.clone().slice(in_ranges.clone());
            let x_im = x_ft_im.clone().slice(in_ranges);

            let (o_re, o_im) = Self::complex_multiplication(
                x_re,
                x_im,
                self.weights_re[corner].val(),
                self.weights_im[corner].val(),
                &self.modes,
                self.flat_modes,
            );

            let mut out_ranges = vec![0..batch, 0..out_ch];
            out_ranges.extend(ranges);
            let out_ranges: [Range<usize>; R] = out_ranges.try_into().unwrap();

            out_ft_re = out_ft_re.slice_assign(out_ranges.clone(), o_re);
            out_ft_im = out_ft_im.slice_assign(out_ranges, o_im);
        }

        Self::ifft_ctensor(out_ft_re, out_ft_im, &orig_dims)
    }
}
