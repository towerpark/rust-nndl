use std::iter;

use ndarray::{s, Axis, Array1, Array2, ArrayView2};
// use ndarray_rand::RandomExt;
use rand::{seq::SliceRandom, thread_rng};

pub type A1 = Array1<f32>;
pub type A2 = Array2<f32>;
pub type V2<'a> = ArrayView2<'a, f32>;
// pub type TrainingData = Vec<(A2, A2)>;
// pub type ValidationData = Vec<(A2, usize)>;

pub struct TrainingData {
    images: A2,
    labels: A2,
}


impl TrainingData {
    pub fn new(images: A2, labels: A2) -> Self {
        TrainingData { images, labels }
    }

    pub fn len(&self) -> usize {
        self.images.len_of(Axis(0))
    }

    // Each sample in a batch is represented by a column vector.
    // pub fn iter(&self, batch_size: usize) -> DataRandomIter<'_> {
    pub fn iter(&self, batch_size: usize) -> impl Iterator<Item = (A2, A2)> + '_ {
        // NOTE:
        //   Create a view with non-contiguous slices is not supported, so we have to copy here.
        //   (https://github.com/rust-ndarray/ndarray/discussions/1050#discussioncomment-1114786)

        let total = self.len();
        let mut indices: Vec<usize> = (0..total).collect();
        indices.shuffle(&mut thread_rng());

        (0..(total / batch_size)).map(move |bi| {
            let begin = bi * batch_size;
            let mut end = begin + batch_size;
            end = if end > total { total } else { end };
            let sample_indices = &indices[begin..end];

            (
                self.images.select(Axis(0), sample_indices),
                self.labels.select(Axis(0), sample_indices),
            )
        })
    }
}

// // pub struct DataRandomIter<'a> {
// pub struct DataRandomIter {
//     images: A2,
//     indices: A2,
// }
// 
// // impl<'a> Iterator for DataRandomIter<'a> {
// impl Iterator for DataRandomIter {
//     // type Item = (ArrayView2<'a, f32>, ArrayView2<'a, f32>);
//     type Item = (A2, A2);
// 
//     fn next(&mut self) -> Option<Self::Item> {
//         todo!();
//     }
// }

pub struct ValidationData {
    images: A2,
    labels: Vec<u8>,
}

impl ValidationData {
    pub fn new(images: A2, labels: Vec<u8>) -> Self {
        ValidationData { images, labels }
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn iter(&self, batch_size: usize) -> impl Iterator<Item = (V2, &[u8])> {
        let total = self.len();
        iter::zip(self.images.outer_iter(), self.labels.iter())
            .enumerate()
            .step_by(batch_size)
            .map(move |(idx, _)| {
                let mut end = idx + batch_size;
                end = if end > total { total } else { end };
                let batched_images = self.images.slice(s![idx..end, ..]);
                let batched_labels = &self.labels[idx..end];
                (batched_images, batched_labels)
            })
    }
}
