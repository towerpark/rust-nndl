use std::time::Instant;

use nndl::{ data_loader, losses::*, network::{ Metrics, Network } };

fn main() {
    let [trn_data, val_data, _] = data_loader::load_mnist("tmp/mnist");
    let mut net = Network::new(vec![784, 30, 10]);

    let mut metrics = Metrics {
        training_loss: Some(Vec::new()),
        training_accuracy: Some(Vec::new()),
        evaluation_loss: Some(Vec::new()),
        evaluation_accuracy: Some(Vec::new()),
    };
    let start_time = Instant::now();
    // Learning rate: 3.0 for MSE loss, 0.5 for cross-entropy loss
    net.sgd::<CrossEntropyLoss>(
        trn_data, 30, 10, 0.5, 5.0, Some(val_data), &mut metrics
    );
    println!("Done: time({:?})", start_time.elapsed());
}
