use super::common::{ A2, V2 };
use super::activations::Activation;

use ndarray::Zip;


pub trait Loss {
    #[allow(dead_code)]
    fn func(a: &A2, y: &V2) -> f32;

    fn delta<N: Activation>(a: A2, y: &V2, z: &A2) -> A2;
}


pub struct CrossEntropyLoss;

impl Loss for CrossEntropyLoss {
    fn func(a: &A2, y: &V2) -> f32 {
        Zip::from(a).and(y).fold(
            0.0, |s, p, q| s + (-q * p.ln() - (1.0 - q) * (1.0 - p).ln())
        )
    }

    fn delta<N: Activation>(a: A2, y: &V2, _: &A2) -> A2 {
        a - y
    }
}


pub struct QuadraticLoss;

impl Loss for QuadraticLoss {
    fn func(a: &A2, y: &V2) -> f32 {
        (a - y).fold(0.0, |s, e| s + e * e) * 0.5
    }

    fn delta<N: Activation>(a: A2, y: &V2, z: &A2) -> A2 {
        (a - y) * N::prime(&z)
    }
}
