use std::time::Instant;

use nndl::{
    data_loader,
    early_stop::EarlyStop,
    losses::*,
    lr_schedulers::*,
    network::{Metrics, Network},
    regularizations::Regularization,
};

fn main() {
    let es_strategy = EarlyStop::unable_to_beat_best(10)
        // EarlyStop::no_inc(5)
        .expect("Early stop strategy should be created successfully.");

    let init_lr = 0.5; // 3.0 for MSE loss and 0.5 for cross-entropy loss
    let lr_schd = AccuracyLrScheduler::new(init_lr, init_lr / 128 as f32, 0.5, 5).unwrap();

    let [trn_data, val_data, _] = data_loader::load_mnist("tmp/mnist");
    let mut net = Network::new(vec![784, 30, 10]);

    let mut metrics = Metrics {
        training_loss: Some(Vec::new()),
        training_accuracy: Some(Vec::new()),
        evaluation_loss: Some(Vec::new()),
        evaluation_accuracy: Some(Vec::new()),
    };
    let start_time = Instant::now();
    net.sgd::<CrossEntropyLoss>(
        trn_data,
        100,
        10,
        lr_schd, // Learning rate: 3.0 for MSE loss, 0.5 for cross-entropy loss
        &Regularization::L2(5.0), // Zero | L1(2.5)
        Some(val_data),
        &mut metrics,
        es_strategy,
    );

    let elapsed = start_time.elapsed();
    let [train_min_loss, eval_min_loss] =
        [metrics.training_loss, metrics.evaluation_loss].map(|m| {
            m.unwrap_or(vec![])
                .into_iter()
                .filter(|l| l.is_finite())
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .map_or("N/A".into(), |v| format!("{:.4}", v))
        });
    let [train_max_acc, eval_max_acc] = [metrics.training_accuracy, metrics.evaluation_accuracy]
        .map(|m| {
            m.unwrap_or(vec![])
                .into_iter()
                .filter(|l| l.is_finite())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .map_or("N/A".into(), |v| format!("{:.2}%", v * 100.0))
        });
    println!(
        "Done: time({:?}), train_min_loss({}), train_max_accuracy({}), \
            eval_min_loss({}), eval_max_accuracy({})",
        elapsed, train_min_loss, train_max_acc, eval_min_loss, eval_max_acc,
    );
}
