use std::time::Instant;

use nndl::{data_loader, network::Network};

fn main() {
    let (trn_data, val_data, _) = data_loader::load_mnist("tmp/mnist");
    let mut net = Network::new(vec![784, 30, 10]);

    let start_time = Instant::now();
    // Learning rate: 3.0 for MSE loss, 0.5 for cross-entropy loss
    net.sgd(trn_data, 30, 10, 0.5, Some(val_data));
    println!("Done: time({:?})", start_time.elapsed());
}

