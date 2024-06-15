use ndarray::Array;
use ndarray_rand::{ rand_distr::{ Normal, StandardNormal }, RandomExt };

use crate::common::A2;


pub trait WeightInitializer {
    fn make_biases(&self, y: usize) -> A2;
    fn make_weights(&self, y: usize, x: usize) -> A2;
}


pub struct DefaultWeightInitializer;

impl WeightInitializer for DefaultWeightInitializer {
    fn make_biases(&self, y: usize) -> A2 {
        Array::random((y, 1), StandardNormal)
    }

    fn make_weights(&self, y: usize, x: usize) -> A2 {
        Array::random(
            (y, x), Normal::new(0.0, 1.0 / (x as f32).sqrt()).unwrap()
        )
    }
}


pub struct LargeWeightInitializer;

impl WeightInitializer for LargeWeightInitializer {
    fn make_biases(&self, y: usize) -> A2 {
        Array::random((y, 1), StandardNormal)
    }

    fn make_weights(&self, y: usize, x: usize) -> A2 {
        Array::random((y, x), StandardNormal)
    }
}
