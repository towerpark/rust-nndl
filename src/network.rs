#![allow(dead_code)] // TODO

use std::iter;

use ndarray::{Axis, Array, ArrayView2, CowArray};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;

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

    fn feedforward(&self, inputs: ArrayView2<f32>) -> A2 {
        let a0 = inputs.t().into_owned();
        iter::zip(self.biases.iter(), self.weights.iter()).fold(
            a0,
            |mut a, (b, w)| {
                a = sigmoid_inplace(w.dot(&a) + b); // b is broadcast
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

            training_data.iter(mini_batch_size).for_each(
                |(samples, truths)| self.update_mini_batch((samples.view(), truths.view()), eta)
            );
            match &test_data {
                Some(data) => {
                    println!(
                        "====== Epoch {}: {} / {} ======",
                        i,
                        self.evaluate(data, mini_batch_size),
                        data.len(),
                    );
                },
                None => println!("====== Epoch {} complete ======", i),
            }
        }
    }

    fn update_mini_batch(&mut self, mini_batch: (ArrayView2<f32>, ArrayView2<f32>), eta: f32) {
        let batch_size = mini_batch.0.len_of(Axis(0));
        let scale = eta / (batch_size as f32);

        let (nabla_b, nabla_w) = self.backprop(mini_batch);

        self.biases.iter_mut().zip(nabla_b).for_each(|(b, nb)| *b -= &(scale * nb));
        self.weights.iter_mut().zip(nabla_w).for_each(|(w, nw)| *w -= &(scale * nw));
    }

    fn backprop(&self, inputs: (ArrayView2<f32>, ArrayView2<f32>)) -> (Vec<A2>, Vec<A2>) {
        let mut nabla_b = Vec::<A2>::new();
        let mut nabla_w = Vec::<A2>::new();
        let samples = inputs.0.t();
        let truths = inputs.1.t();
        let batch_size = samples.len_of(Axis(1));

        // feedforward
        let mut activations = vec![CowArray::from(samples)];
        let mut zs = Vec::<A2>::new();
        for (b, w) in iter::zip(self.biases.iter(), self.weights.iter()) {
            let z = w.dot(activations.last().unwrap()) + b; // b is broadcast
            activations.push(CowArray::from(sigmoid(&z)));
            zs.push(z);
        }

        // backward pass
        let last_delta = Self::cost_derivative(
            activations.last().unwrap().view(), truths
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

        // Sum nabla_b (nabla_w were already summed when performing matrix multiplications above)
        let sum_mat = A2::ones((batch_size, 1));
        nabla_b.iter_mut().for_each(|b| *b = b.dot(&sum_mat));

        (nabla_b.into_iter().rev().collect(), nabla_w.into_iter().rev().collect())
    }

    fn evaluate(&self, test_data: &ValidationData, batch_size: usize) -> usize {
        test_data.iter(batch_size)
            .map(|(samples, truths)| {
                let mut corrected = 0;
                for (output, label) in iter::zip(self.feedforward(samples).columns(), truths) {
                    let idx_of_max_prob = output.t().into_iter()
                        .enumerate()
                        // No Infs and NaNs so we can simply use partial_cmp() here
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx).unwrap();
                    corrected += (idx_of_max_prob == (*label as usize)) as usize;

                }
                corrected
            })
            .sum()
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
