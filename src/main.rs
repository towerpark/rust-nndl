use std::time::Instant;

use nndl::{data_loader, network::Network};

fn main() {
    let (trn_data, val_data, _) = data_loader::load_mnist("tmp/mnist");
    let mut net = Network::new(vec![784, 30, 10]);

    let start_time = Instant::now();
    // epochs = 30, batch_size = 10, lr = 3.0
    net.sgd(trn_data, 30, 10, 3.0, Some(val_data));
    println!("Done: time({:?})", start_time.elapsed());
}

