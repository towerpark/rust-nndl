use super::activations::Activation;
use super::common::{A2, C2};

use ndarray::Zip;

pub trait Loss {
    fn func(outputs: &C2, truths: &C2) -> f32;

    fn delta<N: Activation>(outputs: C2, truths: C2, weighted_inputs: &A2) -> A2;
}

pub struct CrossEntropyLoss;

impl Loss for CrossEntropyLoss {
    fn func(outputs: &C2, truths: &C2) -> f32 {
        Zip::from(outputs).and(truths).fold(0.0, |s, a, y| {
            s + (-y * a.ln() - (1.0 - y) * (1.0 - a).ln())
        })
    }

    fn delta<N: Activation>(outputs: C2, truths: C2, _: &A2) -> A2 {
        outputs.into_owned() - truths
    }
}

pub struct QuadraticLoss;

impl Loss for QuadraticLoss {
    fn func(outputs: &C2, truths: &C2) -> f32 {
        let diff = outputs - truths;
        (&diff * &diff).sum() * 0.5
    }

    fn delta<N: Activation>(outputs: C2, truths: C2, weighted_inputs: &A2) -> A2 {
        (outputs.into_owned() - truths) * N::prime(&weighted_inputs)
    }
}
