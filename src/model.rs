//! 1D Fourier Neural Operator, fixed to the architecture `model.bpk` was
//! trained with. Copied from `sciml_rs::neural_operators::models::fno` for
//! inference only.
//!
//! The training crate carries this as a generic `FNOConfig` + `FNO<const R:
//! usize>` so one config can produce Burgers (`R = 3`), Darcy (`R = 4`), etc.
//! This demo only ever needs the trained Burgers architecture, so the config
//! layer is dropped in favour of the fixed constants below - one less thing to
//! keep in sync by hand between this crate and a run directory.

use alloc::vec::Vec;

use burn::{
    Tensor,
    module::Module,
    nn::{
        Linear, LinearConfig,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{Device, activation::relu},
};

use crate::spectral_conv::SpectralConv;

/// Retained Fourier modes on the spatial axis. Must match `model_cfg.json`'s
/// `modes` for the run `model.bpk` was exported from.
const MODES: usize = 16;
/// Channel width flowing through each spectral layer.
const HIDDEN_CHANNELS: usize = 64;
/// Input channels in the dataset itself (`a(x)`) - the grid coordinate channel
/// is added on top of this, not counted in it. See `FNOConfig::init`.
const DATA_CHANNELS: usize = 1;
/// Number of spectral convolution layers.
const N_LAYERS: usize = 4;

/// Rank 3 = batch + 1 spatial axis + channel axis, fixed for 1D Burgers.
pub type Model = FNO<3>;

#[derive(Module, Debug)]
pub struct FNO<const R: usize> {
    fc0: Linear,
    conv: Vec<SpectralConv<R>>,
    w: Vec<Conv1d>,
    fc1: Linear,
    fc2: Linear,
}

impl FNO<3> {
    /// Builds an untrained model with the architecture `model.bpk` expects.
    /// Weights are loaded over this in [`crate::state::build_and_load_model`].
    pub fn new(device: &Device) -> Self {
        let modes = [MODES];
        let coord_channels = modes.len();

        Self {
            fc0: LinearConfig::new(DATA_CHANNELS + coord_channels, HIDDEN_CHANNELS).init(device),

            conv: (0..N_LAYERS)
                .map(|_| SpectralConv::<3>::new(device, HIDDEN_CHANNELS, HIDDEN_CHANNELS, &modes))
                .collect(),

            w: (0..N_LAYERS)
                .map(|_| Conv1dConfig::new(HIDDEN_CHANNELS, HIDDEN_CHANNELS, 1).init(device))
                .collect(),

            fc1: LinearConfig::new(HIDDEN_CHANNELS, 128).init(device),
            fc2: LinearConfig::new(128, 1).init(device),
        }
    }
}

impl<const R: usize> FNO<R> {
    fn apply_pointwise(conv: &Conv1d, x: Tensor<R>) -> Tensor<R> {
        let dims = x.dims();
        let (b, hidden_channels) = (dims[0], dims[1]);
        let spatial: usize = dims[2..].iter().product();
        conv.forward(x.reshape([b, hidden_channels, spatial]))
            .reshape(dims)
    }

    /// # Shapes
    ///
    /// - input: `[batch, spatial.., data_channels + coord_channels]`
    /// - output: `[batch, spatial.., 1]`
    pub fn forward(&self, x: Tensor<R>) -> Tensor<R> {
        let x = self.fc0.forward(x);

        let perm_in: [usize; R] = core::array::from_fn(|i| match i {
            0 => 0,
            1 => R - 1,
            i => i - 1,
        });
        let mut x = x.permute(perm_in);

        let n = self.conv.len();
        for idx in 0..n {
            let x1 = self.conv[idx].forward(x.clone());
            let x2 = Self::apply_pointwise(&self.w[idx], x);
            x = if idx == n - 1 { x1 + x2 } else { relu(x1 + x2) };
        }

        let perm_out: [usize; R] = core::array::from_fn(|i| match i {
            0 => 0,
            i if i == R - 1 => 1,
            i => i + 1,
        });
        let x = x.permute(perm_out);
        let x = self.fc1.forward(x);
        let x = relu(x);
        self.fc2.forward(x)
    }
}
