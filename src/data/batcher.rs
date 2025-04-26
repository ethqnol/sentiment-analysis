use super::{dataset::IMDBClassificationItem, tokenizer::Tokenizer};
use burn::{
    data::dataloader::batcher::Batcher,
    nn::attention::{GeneratePaddingMask, generate_padding_mask},
    prelude::*,
    tensor::ops::IntElem,
};
use std::sync::Arc;

#[derive(Clone, derive_new::new)]
pub struct IMDBClassificationBatcher<B: Backend> {
    tokenizer: Arc<dyn Tokenizer>,
    device: B::Device,
    max_seq_len: usize,
}

#[derive(Debug, Clone, derive_new::new)]
pub struct IMDBClassificationTraining<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub labels: Tensor<B, 1, Int>,
    pub mask_pad: Tensor<B, 2, Bool>,
}

#[derive(Debug, Clone, derive_new::new)]
pub struct IMDBClassificationInference<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub mask_pad: Tensor<B, 2, Bool>,
}

pub fn pad_to_max_seq_len<B: Backend>(
    pad_token: usize,
    tokens_list: Vec<Vec<usize>>,
    max_seq_length: usize,
    device: &B::Device,
) -> GeneratePaddingMask<B> {
    let batch_size = tokens_list.len();

    let mut tensor = Tensor::zeros([batch_size, max_seq_length], device);
    tensor = tensor.add_scalar(pad_token as i64);

    for (index, tokens) in tokens_list.into_iter().enumerate() {
        let mut seq_length = tokens.len();
        let mut tokens = tokens;

        if seq_length > max_seq_length {
            seq_length = max_seq_length;
            let _ = tokens.split_off(seq_length);
        }

        tensor = tensor.slice_assign(
            [index..index + 1, 0..tokens.len()],
            Tensor::from_data(
                TensorData::new(
                    tokens
                        .into_iter()
                        .map(|e| (e as i64).elem::<IntElem<B>>())
                        .collect(),
                    Shape::new([1, seq_length]),
                ),
                device,
            ),
        );
    }

    let mask = tensor.clone().equal_elem(pad_token as i64);

    GeneratePaddingMask { tensor, mask }
}

impl<B: Backend> Batcher<IMDBClassificationItem, IMDBClassificationTraining<B>>
    for IMDBClassificationBatcher<B>
{
    fn batch(&self, items: Vec<IMDBClassificationItem>) -> IMDBClassificationTraining<B> {
        let mut tokens_list = Vec::with_capacity(items.len());
        let mut labels_list = Vec::with_capacity(items.len());
        for item in items {
            tokens_list.push(self.tokenizer.encode(&item.text));
            labels_list.push(Tensor::from_data(
                TensorData::from([(item.label as i64).elem::<B::IntElem>()]),
                &self.device,
            ));
        }

        let mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokens_list,
            Some(self.max_seq_len),
            &self.device,
        );

        IMDBClassificationTraining {
            tokens: mask.tensor,
            labels: Tensor::cat(labels_list, 0),
            mask_pad: mask.mask,
        }
    }
}


impl<B: Backend> Batcher<String, IMDBClassificationInference<B>> for IMDBClassificationBatcher<B> {
    fn batch(&self, items: Vec<String>) -> IMDBClassificationInference<B> {
        let mut tokens_list = Vec::with_capacity(items.len());

        for item in items {
            tokens_list.push(self.tokenizer.encode(&item));
        }

        let mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokens_list,
            Some(self.max_seq_len),
            &self.device,
        );

        IMDBClassificationInference {
            tokens: mask.tensor.to_device(&self.device),
            mask_pad: mask.mask.to_device(&self.device),
        }
    }
}

