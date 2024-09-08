use std::iter;

use ndarray::Axis;
use serde::{Deserialize, Serialize};

use super::{
    activations::*,
    common::*,
    early_stop::EarlyStop,
    losses::*,
    regularizations::Regularization,
    wb_initializers::*,
};

pub struct Metrics {
    pub training_loss: Option<Vec<f32>>,
    pub training_accuracy: Option<Vec<f32>>,
    pub evaluation_loss: Option<Vec<f32>>,
    pub evaluation_accuracy: Option<Vec<f32>>,
}

#[derive(Serialize, Deserialize)]
pub struct Network {
    sizes: Vec<usize>,
    biases: Vec<A1>,
    weights: Vec<A2>,
}

impl Network {
    pub fn new(sizes: Vec<usize>) -> Self {
        type WBInit = WBInitializerDefault;
        let biases = (&sizes[1..])
            .iter()
            .map(|&s| WBInit::make_biases(s))
            .collect();
        let weights = (&sizes[..(sizes.len() - 1)])
            .iter()
            .zip((&sizes[1..]).iter())
            .map(|(&x, &y)| WBInit::make_weights(y, x))
            .collect();

        Network {
            sizes,
            biases,
            weights,
        }
    }

    pub fn sgd<L>(
        &mut self,
        training_data: Dataset,
        epochs: usize,
        mini_batch_size: usize,
        eta: f32,
        reg: &Regularization,
        evaluation_data: Option<Dataset>,
        metrics: &mut Metrics,
        mut es_strategy: EarlyStop,
    ) where
        L: Loss,
    {
        let data_size = training_data.len();
        let num_of_batches = data_size / mini_batch_size;
        println!("Number of training samples: {}", data_size);
        println!("Number of mini-batches: {}", num_of_batches);
        let mut i = 0usize;
        while i < epochs && !es_strategy.should_stop(metrics) {
            println!("====== Epoch {} started ======", i);

            training_data.iter(mini_batch_size, true).for_each(|batch| {
                self.update_mini_batch::<L>(batch, eta, reg, training_data.len())
            });
            println!("Training complete");

            // Calculate metrics
            if let Some(ref mut tl) = metrics.training_loss {
                let loss = self.total_loss::<L>(&training_data, reg, mini_batch_size);
                tl.push(loss);
                println!("Loss on training data: {:.4}", loss);
            }
            if let Some(ref mut ta) = metrics.training_accuracy {
                let acc = self.accuracy(&training_data, mini_batch_size);
                ta.push(acc);
                println!("Accuracy on training data: {:.2}%", 100.0 * acc);
            }
            if let Some(ref eval_data) = evaluation_data {
                if let Some(ref mut el) = metrics.evaluation_loss {
                    let loss = self.total_loss::<L>(eval_data, reg, mini_batch_size);
                    el.push(loss);
                    println!("Loss on evaluation data: {:.4}", loss);
                }
                if let Some(ref mut ea) = metrics.evaluation_accuracy {
                    let acc = self.accuracy(&eval_data, mini_batch_size);
                    ea.push(acc);
                    println!("Accuracy on evaluation data: {:.2}%", 100.0 * acc);
                }
            }

            println!("====== Epoch {} done ======\n", i);
            i += 1;
        }

        if i < epochs {
            println!("====== Early stopped ======");
        }
    }

    fn update_mini_batch<L>(
        &mut self,
        mini_batch: [C2; 2],
        eta: f32,
        reg: &Regularization,
        n: usize,
    ) where
        L: Loss,
    {
        // Batch size is the length of axis 0 because inputs are column vectors
        let batch_size = mini_batch[0].len_of(Axis(1));
        let scale = eta / (batch_size as f32);

        let (nabla_b, nabla_w) = self.backprop::<Sigmoid, L>(mini_batch);

        self.biases
            .iter_mut()
            .zip(nabla_b)
            .for_each(|(b, nb)| *b -= &(scale * nb));
        self.weights.iter_mut().zip(nabla_w).for_each(|(w, nw)| {
            *w = &*w - eta * (nw / batch_size as f32 + reg.extra_gradient(n, w))
        });
    }

    // Return the gradients of weights and biases for all the layers computed
    // with all the samples in the batch.
    // A weight gradient for a single layer is a JxK matrix, while a biase
    // graident is a J-sized column vector.
    fn backprop<N, L>(&self, inputs: [C2; 2]) -> (Vec<A1>, Vec<A2>)
    where
        N: Activation,
        L: Loss,
    {
        let mut nabla_b = Vec::<A2>::new();
        let mut nabla_w = Vec::<A2>::new();
        let [samples, truths] = inputs;

        // feedforward
        //
        // Both activations and weighted inputs are kept with an JxN matrix,
        // where:
        //   J is the number of neurons in the layer
        //   N is the batch size
        let mut activations = vec![samples];
        let mut zs = Vec::<A2>::new();
        for (b, w) in iter::zip(self.biases.iter(), self.weights.iter()) {
            let z = Self::make_weighted_inputs(&activations.last().unwrap(), w, b);
            activations.push(C2::from(N::call(&z)));
            zs.push(z);
        }

        // backward pass
        //
        // A JxN matrix
        let last_delta = L::delta::<N>(activations.pop().unwrap(), truths, zs.last().unwrap());
        // Activation's size is KxN, so weight graident has a size of JxN * NxK => JxK
        //   Note:
        //     Not only does the matrix multiplication compute gradients with
        //     all the samples at the same time, it also sum the resulting
        //     gradients for every element of the weight matrix.
        nabla_w.push(last_delta.dot(&activations.pop().unwrap().reversed_axes()));
        nabla_b.push(last_delta);
        for l in 2..self.num_layers() {
            let z = &zs[zs.len() - l];
            let sp = N::prime(z);
            let delta = self.weights[self.weights.len() - l + 1]
                .t()
                .dot(nabla_b.last().unwrap())
                * sp;
            nabla_w.push(delta.dot(&activations.pop().unwrap().reversed_axes()));
            nabla_b.push(delta);
        }

        // Sum nabla_b (nabla_w were already summed when performing matrix multiplications above)
        (
            nabla_b
                .iter()
                .map(|nb| nb.sum_axis(Axis(1)))
                .rev()
                .collect(),
            nabla_w.into_iter().rev().collect(),
        )
    }

    fn make_weighted_inputs(inputs: &C2, weights: &A2, biases: &A1) -> A2 {
        // Turn biases into a column vector and broadcast it
        weights.dot(inputs) + biases.to_shape((biases.len(), 1)).unwrap()
    }

    fn num_layers(&self) -> usize {
        self.sizes.len()
    }

    fn total_loss<L>(&self, dataset: &Dataset, reg: &Regularization, batch_size: usize) -> f32
    where
        L: Loss,
    {
        let vanilla_loss = dataset
            .iter(batch_size, false)
            .map(|[images, labels]| {
                let outputs = self.feedforward::<Sigmoid>(images);
                L::func(&outputs, &labels)
            })
            .sum::<f32>();
        vanilla_loss / dataset.len() as f32 + reg.extra_loss(dataset.len(), &self.weights)
    }

    fn accuracy(&self, dataset: &Dataset, batch_size: usize) -> f32 {
        dataset
            .iter(batch_size, false)
            .map(|[images, labels]| {
                let outputs = self.feedforward::<Sigmoid>(images);
                let preds = outputs.columns().into_iter().map(|c| {
                    c.into_iter()
                        .enumerate()
                        // No Infs and NaNs so we can simply use partial_cmp() here
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap()
                });
                let truths = labels
                    .columns()
                    .into_iter()
                    .map(|c| c.into_iter().position(|&e| 1.0 == e).unwrap());
                iter::zip(preds, truths)
                    .map(|(p, t)| (p == t) as usize)
                    .sum::<usize>()
            })
            .sum::<usize>() as f32 / dataset.len() as f32
    }

    fn feedforward<'a, N: Activation>(&self, inputs: C2<'a>) -> C2<'a> {
        iter::zip(self.biases.iter(), self.weights.iter()).fold(inputs, |a, (b, w)| {
            C2::from(N::call(&Self::make_weighted_inputs(&a, w, b)))
        })
    }
}
