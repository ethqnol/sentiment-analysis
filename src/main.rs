#![allow(dead_code)]
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::{
    backend::Autodiff,
    data::dataloader::batcher::Batcher,
    module::Module,
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
};
use sentiment_analysis::data::{
    IMDBClassificationBatcher, IMDBClassificationInference, IMDBClassificationItem,
};
use sentiment_analysis::{
    data::{BertCasedTokenizer, Tokenizer},
    model::Model,
    train,
};
use std::io::{Write, stdin, stdout};
use std::sync::Arc;

pub type WgpuBackend = Wgpu<f32, i32>;

fn main() {
    run_train();

    use std::env;
    use std::path::PathBuf;

    let tokenizer = Arc::new(BertCasedTokenizer::default());

    type WgpuBackend = Wgpu<f32, i32>;
    let model: Model<WgpuBackend> = Model::new(&Default::default(), tokenizer.vocab_size());
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = model
        .load_file(format!("./model/model"), &recorder, &WgpuDevice::default())
        .expect("Model failed to load");

    let batcher: IMDBClassificationBatcher<WgpuBackend> =
        IMDBClassificationBatcher::<WgpuBackend>::new(tokenizer.clone(), Default::default(), 256);

    let mut analysis = String::new();
    print!("Enter comment: ");
    let _ = stdout().flush();
    stdin()
        .read_line(&mut analysis)
        .expect("Did not enter a correct string");

    let items: Vec<String> = vec![analysis];

    let item = batcher.batch(items);

    let output = model.forward(item.tokens, item.mask_pad);

    println!("{}", burn::tensor::activation::softmax(output, 1))
}

fn run_train() {
    type AutodiffBackend = Autodiff<WgpuBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    train::run::<AutodiffBackend>(device.clone());
}
