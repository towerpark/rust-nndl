use std::cmp;

use ndarray::{s, Axis, Array1, Array2, CowArray, Ix2};
// use ndarray_rand::RandomExt;
use rand::{seq::SliceRandom, thread_rng};

pub type A1 = Array1<f32>;
pub type A2 = Array2<f32>;
pub type C2<'a> = CowArray<'a, f32, Ix2>;


// Data is kept in row vectors
pub struct Dataset {
    images: A2,
    labels: A2,
}


impl Dataset {
    pub fn new(images: A2, labels: Vec<u8>) -> Self {
        Self {
            images,
            labels: Self::vectorized_result(&labels),
        }
    }

    pub fn len(&self) -> usize {
        self.images.len_of(Axis(0))
    }

    // Each sample in a batch is represented by a column vector.
    // pub fn iter(&self, batch_size: usize) -> DataRandomIter<'_> {
    pub fn iter<'a>(
        &'a self, batch_size: usize, shuffle: bool
    ) -> impl Iterator<Item = [C2; 2]> {
        let total = self.len();
        let sampler: Box<dyn Fn(usize) -> [C2<'a>; 2]>;

        if shuffle {
            let mut indices: Vec<usize> = (0..total).collect();
            indices.shuffle(&mut thread_rng());
            sampler = Box::new(move |begin| {
                let end = cmp::min(begin + batch_size, total);
                let sample_indices = &indices[begin..end];
                // NOTE:
                //   Create a view with non-contiguous slices is not supported,
                //   so we have to copy here.
                //   (https://github.com/rust-ndarray/ndarray/discussions/1050#discussioncomment-1114786)
                [
                    C2::from(self.images.select(Axis(0), sample_indices)),
                    C2::from(self.labels.select(Axis(0), sample_indices)),
                ]
            });
        }
        else {
            sampler = Box::new(move |begin| {
                let end = cmp::min(begin + batch_size, total);
                [
                    C2::from(self.images.slice(s![begin..end, ..])),
                    C2::from(self.labels.slice(s![begin..end, ..])),
                ]
            });
        }

        (0..total).step_by(batch_size).map(
            move |i| sampler(i).map(C2::reversed_axes)
        )
    }

    fn vectorized_result(labels: &Vec<u8>) -> A2 {
        A2::from_shape_fn(
            (labels.len(), 10),
            |(m, n)| (n == labels[m] as usize) as i32 as f32,
        )
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
