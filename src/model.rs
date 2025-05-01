use burn::{
    module::Module,
    nn::{
        loss::CrossEntropyLossConfig, BiLstm, BiLstmConfig, Dropout, DropoutConfig, Embedding, EmbeddingConfig, Initializer::XavierUniform, LayerNorm, LayerNormConfig, Linear, LinearConfig
    },
    prelude::*,
    tensor::{
        backend::{AutodiffBackend, Backend}, Tensor
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::data::IMDBClassificationTraining;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    lstm: BiLstm<B>,
    linear: Linear<B>,
    dropout: Dropout,
    layer_norm: LayerNorm<B>,
    embedding: Embedding<B>,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device, vocab_size: usize) -> Self {
        let embed_dim: usize = 512;
        Self {
            lstm: BiLstmConfig::new(512, 512, true).init(device),
            linear: LinearConfig::new(512*2, 2).with_initializer(XavierUniform { gain: 1.0 }).init(device),
            dropout: DropoutConfig::new(0.3).init(),
            layer_norm: LayerNormConfig::new(512*2).init(device),
            embedding: EmbeddingConfig::new(vocab_size, embed_dim).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>, mask_pad: Tensor<B, 2, Bool>) -> Tensor<B, 2> {
        let embedded_tokens = self.embedding.forward(input.clone());
        
        let (_output, lstm_state) = self.lstm.forward(embedded_tokens, None);
        
        //let [batch_size, seq_len, hidden_size] = _output.dims();
        let hidden = lstm_state.hidden; 
        let [directions_layers, batch_size, hidden_size] = hidden.dims();
        let num_layers = directions_layers / 2;
        
        
        let hidden_forward = hidden.clone().slice([2 * (num_layers - 1)..2 * (num_layers - 1) + 1, 0..batch_size, 0..hidden_size]).squeeze::<2>(0); 
        let hidden_backward = hidden.clone().slice([2 * (num_layers - 1) + 1..2 * (num_layers - 1) + 2, 0..batch_size, 0..hidden_size]).squeeze::<2>(0);
    
        let hidden = Tensor::cat(vec![hidden_forward, hidden_backward], 1);
        let output = self.layer_norm.forward(hidden);
        

        let output = self.dropout.forward(output);
        let output = self.linear.forward(output);

        return output;
    }
}

impl<B: AutodiffBackend> TrainStep<IMDBClassificationTraining<B>, ClassificationOutput<B>>
    for Model<B>
{
    fn step(&self, item: IMDBClassificationTraining<B>) -> TrainOutput<ClassificationOutput<B>> {
        let tokens = item.tokens;
        let targets = item.labels;
        let mask_pad = item.mask_pad;
        let output = self.forward(tokens, mask_pad);

        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        let item = ClassificationOutput {
            loss,
            output,
            targets,
        };
        return TrainOutput::new(self, item.loss.backward(), item);
    }
}

impl<B: Backend> ValidStep<IMDBClassificationTraining<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: IMDBClassificationTraining<B>) -> ClassificationOutput<B> {
        let tokens = item.tokens;
        let mask_pad = item.mask_pad;
        let targets = item.labels;
        let output = self.forward(tokens, mask_pad);

        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        return ClassificationOutput {
            loss,
            output,
            targets,
        };

    }
}







// == ignore ==
// let idx = mask_pad
//     .bool_not()
//     .flip([1])
//     .int()
//     .argmax(1)
//     .neg()
//     .add_scalar(seq_len as i32 - 1)
//     .unsqueeze_dim::<3>(1)
//     .expand([batch_size, 1, hidden_size]);

// let output = output.gather(1, idx);
// let output = output.squeeze::<2>(1);
//let output = output.slice([0..batch_size, seq_len - 1..seq_len, 0..hidden_size]).squeeze::<2>(1);