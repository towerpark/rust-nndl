use super::common::A2;

pub enum Regularization {
    L1(f32),
    L2(f32),
    Zero, // Unregularized
}

impl Regularization {
    pub fn extra_loss(&self, dataset_size: usize, weights: &Vec<A2>) -> f32 {
        match self {
            Self::L1(lmbda) => {
                lmbda * weights.iter().map(|w| w.abs().sum()).sum::<f32>() / dataset_size as f32
            }
            Self::L2(lmbda) => {
                0.5 * lmbda * weights.iter().map(|w| (w * w).sum()).sum::<f32>()
                    / dataset_size as f32
            }
            Self::Zero => 0.0,
        }
    }

    pub fn extra_gradient(&self, dataset_size: usize, weights: &A2) -> A2 {
        match self {
            &Self::L1(lmbda) => lmbda / dataset_size as f32 * weights.signum(),
            &Self::L2(lmbda) => lmbda / dataset_size as f32 * weights,
            Self::Zero => A2::zeros(weights.raw_dim()),
        }
    }
}
