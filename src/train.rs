use crate::{
    data::{BertCasedTokenizer, IMDBClassificationBatcher, IMDBDataset, Tokenizer},
    model::Model,
};
use burn::{
    data::dataloader::DataLoaderBuilder,
    lr_scheduler::noam::NoamLrSchedulerConfig,
    optim::AdamConfig,
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{
        LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
        metric::{
            AccuracyMetric, LossMetric,
            store::{Aggregate, Direction, Split},
        },
    },
};
use std::sync::Arc;

static ARTIFACT_DIR: &str = "./model";

#[derive(Config)]
pub struct IMDBTrainingConfig {
    #[config(default = 7)]
    pub num_epochs: usize,

    #[config(default = 64)]
    pub batch_size: usize,

    #[config(default = 42)]
    pub seed: u64,

    pub optimizer: AdamConfig,

    pub vocab_size: usize,

    #[config(default = 256)]
    pub max_seq_len: usize,
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    create_artifact_dir(ARTIFACT_DIR);
    let tokenizer = Arc::new(BertCasedTokenizer::default());
    let config_optimizer = AdamConfig::new();
    let config = IMDBTrainingConfig::new(config_optimizer, tokenizer.vocab_size());
    let lr_scheduler = NoamLrSchedulerConfig::new(1e-4)
            .with_warmup_steps(2500)
            .with_model_size(516)
            .init()
            .unwrap();
    let batcher_train =
        IMDBClassificationBatcher::<B>::new(tokenizer.clone(), device.clone(), config.max_seq_len);

    let batcher_test = IMDBClassificationBatcher::<B::InnerBackend>::new(
        tokenizer.clone(),
        device.clone(),
        config.max_seq_len,
    );

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(IMDBDataset::train());
    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(IMDBDataset::test());
    B::seed(config.seed);
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 1 },
        ))
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            Model::<B>::new(&device, tokenizer.vocab_size()),
            config.optimizer.init(),
            lr_scheduler,
        );

    let trained = learner.fit(dataloader_train, dataloader_test);
    config.save(format!("{ARTIFACT_DIR}/config.json")).unwrap();
    CompactRecorder::new()
        .record(
            trained.into_record(),
            format!("{ARTIFACT_DIR}/model").into(),
        )
        .unwrap();
}
