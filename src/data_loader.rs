use ndarray::Array2;

use mnist::*;

use super::common::Dataset;

// TODO: data_dir: &str => Path
pub fn load_mnist(data_dir: &str) -> [Dataset; 3] {
    const TRAINING_SET_SIZE: usize = 50_000;
    const VALIDATION_SET_SIZE: usize = 10_000;
    const TEST_SET_SIZE: usize = 10_000;

    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        val_img,
        val_lbl,
        tst_img,
        tst_lbl,
    } = MnistBuilder::new()
        .base_path(data_dir)
        .label_format_digit()
        .training_set_length(TRAINING_SET_SIZE as u32)
        .validation_set_length(VALIDATION_SET_SIZE as u32)
        .test_set_length(TEST_SET_SIZE as u32)
        .finalize();

    let image_info = [
        (trn_img, TRAINING_SET_SIZE, "training"),
        (val_img, VALIDATION_SET_SIZE, "validation"),
        (tst_img, TEST_SET_SIZE, "test"),
    ];
    let image_sets = image_info.map(|(blob, num_of_imgs, name)| {
        Array2::from_shape_vec((num_of_imgs, 28 * 28), blob)
            .expect(std::format!("Error converting {} set images", name).as_str())
            .map(|x| (*x as f32) / 255.0_f32)
    });

    let [trn_imgs, val_imgs, tst_imgs] = image_sets;

    [
        Dataset::new(trn_imgs, trn_lbl),
        Dataset::new(val_imgs, val_lbl),
        Dataset::new(tst_imgs, tst_lbl),
    ]
}
