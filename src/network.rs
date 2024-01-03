#![allow(dead_code)] // TODO

use std::iter;

use ndarray::{Axis, Array, ArrayView2, CowArray};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use rand::{seq::SliceRandom, thread_rng};

use super::common::*;

pub struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    biases: Vec<A2>,
    weights: Vec<A2>,
}


impl Network {
    pub fn new(sizes: Vec<usize>) -> Self {
        let biases = (&sizes[1..]).iter().map(|&s| {
            Array::random((s, 1), StandardNormal)
        }).collect();
        let weights = (&sizes[..(sizes.len() - 1)]).iter().zip((&sizes[1..]).iter()).map(
            |(&x, &y)| Array::random((y, x), StandardNormal)
        ).collect();

        Network {
            num_layers: sizes.len(),
            sizes,
            biases,
            weights,
        }
    }

    fn feedforward(&self, input: ArrayView2<f32>) -> A2 {
        let a0 = input.into_owned();
        iter::zip(self.biases.iter(), self.weights.iter()).fold(
            a0,
            |mut a, (b, w)| {
                a = sigmoid_inplace(w.dot(&a) + b);
                a
            }
        )
    }

    pub fn sgd(
        &mut self,
        training_data: TrainingData,
        epochs: usize,
        mini_batch_size: usize,
        eta: f32,
        test_data: Option<ValidationData>,
    ) {
        let data_size = training_data.len();
        let num_of_batches = data_size / mini_batch_size;
        println!("Number of training samples: {}", data_size);
        println!("Number of mini-batches: {}", num_of_batches);
        for i in 0..epochs {
            println!("====== Epoch {} started ======", i);

            let mut indices: Vec<usize> = (0..data_size).collect();
            indices.shuffle(&mut thread_rng());
            for k in 0_usize..num_of_batches {
                // println!("\tBatch {} ...", k);

                let batch: Vec<(ArrayView2<f32>, ArrayView2<f32>)> = (
                    indices[k * mini_batch_size..(k + 1) * mini_batch_size]
                ).iter().map(|slot| training_data.elem(*slot)).collect();
                self.update_mini_batch(batch, eta);
            }
            match &test_data {
                Some(data) => {
                    let n_test = data.len();
                    println!("====== Epoch {}: {} / {} ======", i, self.evaluate(data), n_test);
                },
                None => println!("====== Epoch {} complete ======", i),
            }
        }
    }

    fn update_mini_batch(&mut self, mini_batch: Vec<(ArrayView2<f32>, ArrayView2<f32>)>, eta: f32) {
        let mut nabla_b: Vec<A2> = self.biases.iter().map(|b| Array::zeros(b.raw_dim())).collect();
        let mut nabla_w: Vec<A2> = self.weights.iter().map(|w| Array::zeros(w.raw_dim())).collect();
        let batch_size = mini_batch.len();
        for (sample, truth) in mini_batch {
            let (delta_nabla_b, delta_nabla_w) = self.backprop(sample, truth);
            nabla_b = iter::zip(nabla_b, delta_nabla_b).map(|(nb, dnb)| nb + dnb).collect();
            nabla_w = iter::zip(nabla_w, delta_nabla_w).map(|(nw, dnw)| nw + dnw).collect();
        }
        let scale = eta / (batch_size as f32);
        self.biases.iter_mut().zip(nabla_b).for_each(|(b, nb)| *b -= &(scale * nb));
        self.weights.iter_mut().zip(nabla_w).for_each(|(w, nw)| *w -= &(scale * nw));
    }

    fn backprop(&self, input: ArrayView2<f32>, ground_truth: ArrayView2<f32>) -> (Vec<A2>, Vec<A2>) {
        let mut nabla_b = Vec::<A2>::new();
        let mut nabla_w = Vec::<A2>::new();

        // feedforward
        let mut activations = vec![CowArray::from(input)];
        let mut zs = Vec::<A2>::new();
        for (b, w) in iter::zip(self.biases.iter(), self.weights.iter()) {
            let z = w.dot(activations.last().unwrap()) + b;
            activations.push(CowArray::from(sigmoid(&z)));
            zs.push(z);
        }

        // backward pass
        let last_delta = Self::cost_derivative(
            activations.last().unwrap().view(), ground_truth
        ) * sigmoid_prime(zs.last().unwrap());
        nabla_w.push(last_delta.dot(&activations[activations.len() - 2].t()));
        nabla_b.push(last_delta);
        for l in 2..self.num_layers {
            let z = &zs[zs.len() - l];
            let sp = sigmoid_prime(z);
            let delta = self.weights[self.weights.len() - l + 1].t().dot(nabla_b.last().unwrap()) * sp;
            nabla_w.push(delta.dot(&activations[activations.len() - l - 1].t()));
            nabla_b.push(delta);
        }

        (nabla_b.into_iter().rev().collect(), nabla_w.into_iter().rev().collect())
    }

    fn evaluate(&self, test_data: &ValidationData) -> usize {
        test_data.iter().map(|(x, y)| {
            let output = self.feedforward(x).remove_axis(Axis(1));
            let idx_of_max_prob = output.into_iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx).unwrap();
            (idx_of_max_prob == y) as usize
        }).sum()
    }

    fn cost_derivative(output_activations: ArrayView2<f32>, y: ArrayView2<f32>) -> A2 {
        output_activations.into_owned() - y
    }
}


fn sigmoid_scalar(z: f32) -> f32 {
    1.0f32 / (1.0f32 + std::f32::consts::E.powf(-z))
}

fn sigmoid(z: &A2) -> A2 {
    z.mapv(|e| sigmoid_scalar(e))
}

fn sigmoid_inplace(mut z: A2) -> A2 {
    z.map_inplace(|e| *e = sigmoid_scalar(*e));
    z
}

fn sigmoid_prime(z: &A2) -> A2 {
    let mut s = sigmoid(z);
    s.map_inplace(|e| *e *= 1.0_f32 - *e);
    s
}
