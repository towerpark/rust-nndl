use crate::network::Metrics;

// The number of epoches is 0-based.
//
// "window_size" is not extracted and shared between the two strategies because
// we migth add a complete different strategy in future that won't need such a
// field (e.g., one that predicts the possibility of further improvement based
// on history accuracies).
pub enum EarlyStop {
    // No improvement in N epochs
    UnableToBeatBest {
        window_size: usize,
        best_epoch: usize,
    },
    // Train N more epochs on each improvement
    NoInc {
        window_size: usize,
        stop_at: usize,
    },
}

impl EarlyStop {
    pub fn unable_to_beat_best(window_size: usize) -> Result<Self, String> {
        if 0 == window_size {
            return Err("Window size must be greater than 0!".into());
        }
        Ok(Self::UnableToBeatBest {
            window_size,
            best_epoch: 0,
        })
    }

    pub fn no_inc(window_size: usize) -> Result<Self, String> {
        if 0 == window_size {
            return Err("Window size must be greater than 0!".into());
        }
        Ok(Self::NoInc {
            window_size,
            stop_at: window_size,
        })
    }

    pub fn should_stop(&mut self, metrics: &Metrics) -> bool {
        // TODO: Use evaluation loss instead if evaluation accuracy is unavailable.
        let acc;
        match &metrics.evaluation_accuracy {
            Some(ea) if !ea.is_empty() => acc = ea,
            _ => return false,
        }

        let cur_epoch = acc.len() - 1;
        match self {
            Self::UnableToBeatBest {window_size, best_epoch} => {
                if acc[cur_epoch] > acc[*best_epoch] {
                    *best_epoch = cur_epoch;
                }
                cur_epoch - *best_epoch == *window_size
            },
            Self::NoInc {window_size, stop_at} => {
                if cur_epoch >= 1 && acc[cur_epoch] > acc[cur_epoch - 1] {
                    *stop_at = cur_epoch + *window_size;
                }
                cur_epoch == *stop_at
            }
        }
    }
}

