#![allow(clippy::new_without_default)]

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

use crate::model::Model;
use crate::state::build_and_load_model;

use burn::tensor::{Device, Tensor, signal};

/// Fourier modes a freehand-drawn initial condition is projected onto before
/// prediction. A mouse-drawn curve is far outside the smooth-GRF training
/// distribution (sharp corners, pixel jitter); low-pass filtering it first is
/// what keeps the prediction from looking broken. Presets and slider-generated
/// ICs are already in-distribution and skip this - see `project_to_k_modes`.
const IC_PROJECTION_MODES: usize = 16;

#[cfg_attr(target_family = "wasm", wasm_bindgen(start))]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Low-pass-projects `ic` onto its first [`IC_PROJECTION_MODES`] Fourier
/// modes, at `ic`'s own length. Stateless - call this on a freehand-drawn
/// curve before [`Fno1d::predict`]; preset and slider-generated ICs need it.
///
/// # Arguments
///
/// * `ic` - samples of length a power of two.
#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub async fn project_to_k_modes(ic: &[f32]) -> Result<Box<[f32]>, String> {
    assert!(
        ic.len().is_power_of_two(),
        "ic length must be a power of two, got {}",
        ic.len()
    );

    let device: Device = Default::default();
    let x = Tensor::<1>::from_floats(ic, &device);

    let (re, im) = signal::rfft(x, 0, None);
    let k = IC_PROJECTION_MODES.min(re.dims()[0]);
    let re = re.narrow(0, 0, k);
    let im = im.narrow(0, 0, k);
    let projected = signal::irfft(re, im, 0, Some(ic.len()));

    let data = projected.into_data_async().await.unwrap();
    Ok(data.iter::<f32>().collect::<Vec<f32>>().into_boxed_slice())
}

/// Resamples `ic` to `n_eval` samples by zero-padding (or truncating) its
/// spectrum - not interpolation. `signal::irfft`'s `n` argument does this
/// directly: fewer input bins than `n_eval` implies are zero-padded, more are
/// truncated, in both cases in frequency space.
///
/// `rfft` is unnormalized (`X[k] = Σ x[n] exp(...)`) and `irfft` divides by
/// its own output length `n_eval`, not by `ic.len()` the forward transform
/// used - so reconstructing at a different length than the input needs a
/// `n_eval / ic.len()` correction, or amplitude shrinks (grows) with every
/// upsample (downsample). Standard FFT-interpolation scaling (e.g.
/// `interpft`), not a Burn-specific quirk.
fn resample_spectral(ic: &[f32], n_eval: usize, device: &Device) -> Tensor<1> {
    let x = Tensor::<1>::from_floats(ic, device);
    let (re, im) = signal::rfft(x, 0, None);
    let resampled = signal::irfft(re, im, 0, Some(n_eval));
    resampled.mul_scalar(n_eval as f64 / ic.len() as f64)
}

/// Uniform grid on `[0, 1]` with `n` points, matching
/// `sciml_rs::neural_operators::data::grids::uniform_grid_1d` (an inclusive
/// linspace) - generated fresh at every call, never cached from a fixed
/// resolution.
fn uniform_grid_1d(n: usize, device: &Device) -> Tensor<1> {
    let step = 1.0 / (n - 1) as f32;
    let coords: Vec<f32> = (0..n).map(|i| i as f32 * step).collect();
    Tensor::<1>::from_floats(coords.as_slice(), device)
}

/// `Fno1d` structure that corresponds to a JavaScript class.
/// See: [exporting-rust-struct](https://rustwasm.github.io/wasm-bindgen/contributing/design/exporting-rust-struct.html)
#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub struct Fno1d {
    model: Option<Model>,
}

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
impl Fno1d {
    /// Constructor called by JavaScript with the `new` keyword.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        Self { model: None }
    }

    /// Predicts `u(x, T)` at `n_eval` samples from an initial condition `ic`.
    ///
    /// This method is called from JavaScript via generated wrapper code by wasm-bindgen.
    ///
    /// # Arguments
    ///
    /// * `ic` - initial condition samples, length a power of two. A
    ///   freehand-drawn curve must already have been passed through
    ///   [`project_to_k_modes`] before it reaches here.
    /// * `n_eval` - output resolution, a power of two. May differ from
    ///   `ic.len()` in either direction - this is the resolution slider.
    pub async fn predict(&mut self, ic: &[f32], n_eval: usize) -> Result<Box<[f32]>, String> {
        assert!(
            ic.len().is_power_of_two(),
            "ic length must be a power of two, got {}",
            ic.len()
        );
        assert!(
            n_eval.is_power_of_two(),
            "n_eval must be a power of two, got {n_eval}"
        );

        if self.model.is_none() {
            self.model = Some(build_and_load_model().await);
        }
        let model = self.model.as_ref().unwrap();

        let device: Device = Default::default();

        // Reproduce training-time input assembly exactly: channel 0 is a(x)
        // resampled to n_eval, channel 1 is a grid generated fresh at n_eval -
        // never reused from training resolution, or the resolution slider
        // silently breaks.
        let a = resample_spectral(ic, n_eval, &device).reshape([n_eval, 1]);
        let grid = uniform_grid_1d(n_eval, &device).reshape([n_eval, 1]);
        let input = Tensor::cat(alloc::vec![a, grid], 1).reshape([1, n_eval, 2]);

        assert_eq!(
            input.dims(),
            [1, n_eval, 2],
            "assembled input must match n_eval"
        );

        // Run the tensor input through the model.
        let output: Tensor<3> = model.forward(input);
        let output = output.reshape([n_eval]);

        let data = output.into_data_async().await.unwrap();
        Ok(data.iter::<f32>().collect::<Vec<f32>>().into_boxed_slice())
    }
}
