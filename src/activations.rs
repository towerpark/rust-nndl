use super::common::A2;

pub trait Activation {
    fn call(z: &A2) -> A2;

    fn prime(z: &A2) -> A2;
}

pub struct Sigmoid;

impl Activation for Sigmoid {
    fn call(z: &A2) -> A2 {
        z.mapv(|e| 1.0f32 / (1.0f32 + std::f32::consts::E.powf(-e)))
    }

    fn prime(z: &A2) -> A2 {
        let mut s = Self::call(&z);
        s.map_inplace(|e| *e *= 1.0_f32 - *e);
        s
    }
}
