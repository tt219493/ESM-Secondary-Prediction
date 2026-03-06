import torch
from torch.utils.data import random_split, DataLoader
import lightning as L
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from datasets import Dataset

class EsmDataModule(L.LightningDataModule):
    def __init__(self, train_df, test_df, batch_size = 16, pretrained: str = "facebook/esm2_t6_8M_UR50D"):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer, padding=True)

    def prepare_data(self):
        self.train_ds = Dataset.from_polars(self.train_df.select(['input_ids', 'attention_mask', 'label']).collect())
        self.predict_ds = Dataset.from_polars(self.test_df.collect())
        self.test_ds = self.predict_ds.remove_columns(['id', 'asym_id', 'sequence', 'index'])
    
    def setup(self, stage: str):
        if stage == "fit":
            self.train_ds, self.val_ds = random_split(self.train_ds, [0.8, 0.2], 
                                                      generator=torch.Generator().manual_seed(123))
        if stage == "test":
            self.test_ds = self.test_ds
        
        if stage == "predict":
            self.predict_ds = self.predict_ds
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, collate_fn=self.collator, batch_size=self.batch_size, shuffle=True,
                            num_workers = 2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, collate_fn=self.collator, batch_size=self.batch_size,
                        num_workers = 2)

    def test_dataloader(self):
        return DataLoader(self.test_ds, collate_fn=self.collator, batch_size=self.batch_size,
                        num_workers = 2)

    def predict_dataloader(self):
        return DataLoader(self.predict_ds, batch_size=1,
                        num_workers = 2)





