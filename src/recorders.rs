use std::{fs::File, io, path::Path};

use super::network::Network;

// Trait for saving/loading network weights into/from files
pub trait Recorder {
    fn save(model: &Network, file_path: &Path) -> io::Result<()>;

    fn load(file_path: &Path) -> io::Result<Network>;
}


pub struct JSONRecorder;

impl Recorder for JSONRecorder {
    fn save(model: &Network, file_path: &Path) -> io::Result<()> {
        let file = File::create(file_path)?;
        serde_json::to_writer(file, model)?;
        Ok(())
    }

    fn load(file_path: &Path) -> io::Result<Network> {
        let file = File::open(file_path)?;
        let model = serde_json::from_reader(file)?;
        Ok(model)
    }
}
