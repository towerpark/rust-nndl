use ndarray::Array;
use ndarray_rand::{
    rand_distr::{Normal, StandardNormal},
    RandomExt,
};

use crate::common::{A1, A2};

pub trait WBInitializer {
    fn make_weights(y: usize, x: usize) -> A2;

    fn make_biases(y: usize) -> A1 {
        Array::random(y, StandardNormal)
    }
}

pub struct WBInitializerDefault;

impl WBInitializer for WBInitializerDefault {
    fn make_weights(y: usize, x: usize) -> A2 {
        Array::random((y, x), Normal::new(0.0, 1.0 / (x as f32).sqrt()).unwrap())
    }
}

pub struct WBInitializerLarge;

// Biases and weights are initialized randomly, i.e., they follow Gaussian distributions with mean 0
// and standard deviation 1.
impl WBInitializer for WBInitializerLarge {
    fn make_weights(y: usize, x: usize) -> A2 {
        Array::random((y, x), StandardNormal)
    }
}
