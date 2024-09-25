use crate::network::Metrics;

pub trait LrScheduler {
    // Should be called every epoch with historical accuracies to get LR.
    fn next(&mut self, metrics: &Metrics) -> f32;
}

pub struct AccuracyLrScheduler {
    // Decay ratio
    gamma: f32,
    // Lower bound of LR
    until_lr: f32,
    // Decay LR when there is no improvement in this many epochs
    window_size: usize,

    current_lr: f32,
    // The epoch from which we start to count against "window_size"
    window_begin: usize,
}

impl AccuracyLrScheduler {
    pub fn new(
        init_lr: f32,
        until_lr: f32,
        gamma: f32,
        window_size: usize,
    ) -> Result<Self, String> {
        if window_size == 0 {
            return Err("[AccuracyLrScheduler] Window size must be a positive number!".into());
        }

        if init_lr <= 0.0 || until_lr <= 0.0 {
            println!("[AccuracyLrScheduler] Warning: At least one learning rate is non-positive.");
        }
        if init_lr <= until_lr {
            println!("[AccuracyLrScheduler] Warning: Initial learning rate is less than or equal to the final learning rate.");
        }
        if gamma <= 0.0 || gamma >= 1.0 {
            println!("[AccuracyLrScheduler] Warning: Gamma is out of range (0.0, 1.0).");
        }

        Ok(Self {
            gamma,
            until_lr,
            window_size,
            current_lr: init_lr,
            window_begin: 0,
        })
    }
}

impl LrScheduler for AccuracyLrScheduler {
    fn next(&mut self, metrics: &Metrics) -> f32 {
        if self.current_lr > self.until_lr {
            match &metrics.evaluation_accuracy {
                Some(acc) if !acc.is_empty() => {
                    let cur_epoch = acc.len() - 1;
                    if acc[cur_epoch] > acc[self.window_begin] {
                        self.window_begin = cur_epoch; // Accuracy improved
                    } else if cur_epoch - self.window_begin == self.window_size {
                        self.current_lr *= self.gamma;
                        self.window_begin = cur_epoch;
                    }
                }
                _ => (),
            }
        }
        self.current_lr
    }
}
