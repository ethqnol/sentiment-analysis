use burn::data::dataset::{Dataset, SqliteDataset, source::huggingface::HuggingfaceDatasetLoader};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug)]
pub struct IMDBClassificationItem {
    pub label: usize,
    pub text: String,
}

impl IMDBClassificationItem {
    pub fn new(label: usize, text: String) -> IMDBClassificationItem {
        return Self { label, text };
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IMDBItem {
    pub label: usize,
    pub text: String,
}

pub struct IMDBDataset {
    dataset: SqliteDataset<IMDBItem>,
}

impl Dataset<IMDBClassificationItem> for IMDBDataset {
    fn get(&self, index: usize) -> Option<IMDBClassificationItem> {
        self.dataset
            .get(index)
            .map(|item| IMDBClassificationItem::new(item.label, item.text))
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl IMDBDataset {
    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn test() -> Self {
        Self::new("test")
    }

    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<IMDBItem> = HuggingfaceDatasetLoader::new("imdb")
            .dataset(split)
            .unwrap();
        Self { dataset }
    }
}
