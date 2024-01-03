use std::iter;

use ndarray::{Axis, Array2, ArrayView2, Array3};

pub type A2 = Array2<f32>;
pub type A3 = Array3<f32>;
// pub type TrainingData = Vec<(A2, A2)>;
// pub type ValidationData = Vec<(A2, usize)>;

pub struct TrainingData {
    images: A3,
    labels: A3,
}

impl TrainingData {
    pub fn new(images: A3, labels: A3) -> Self {
        TrainingData { images, labels }
    }

    pub fn len(&self) -> usize {
        self.images.len_of(Axis(0))
    }

    pub fn elem(&self, index: usize) -> (ArrayView2<f32>, ArrayView2<f32>) {
        (
            self.images.index_axis(Axis(0), index),
            self.labels.index_axis(Axis(0), index),
        )
    }
}

pub struct ValidationData {
    images: A3,
    labels: Vec<u8>,
}

impl ValidationData {
    pub fn new(images: A3, labels: Vec<u8>) -> Self {
        ValidationData { images, labels }
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (ArrayView2<f32>, usize)> {
        iter::zip(
            self.images.outer_iter(),
            self.labels.iter().map(|i| *i as usize)
        )
    }
}
