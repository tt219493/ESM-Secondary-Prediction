import torch
from torch.utils.data import random_split, DataLoader
import lightning as L
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from datasets import Dataset

class EsmDataModule(L.LightningDataModule):
    def __init__(self, 
                 train_df, 
                 test_df, 
                 val_df = None, 
                 predict_df = None, 
                 num_workers = 2, 
                 batch_size = 16, 
                 eval_batch_size = 16, 
                 predict_batch_size = 1,
                 pretrained: str = "facebook/esm2_t6_8M_UR50D"):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        self.predict_df = predict_df
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.predict_batch_size = predict_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer, padding=True)
    
    def setup(self, stage: str):
        if stage == "fit":
            self.train_ds = Dataset.from_polars(self.train_df.select(['input_ids', 'attention_mask', 'label']).collect())
            
            if self.val_df is not None:
                self.val_ds = Dataset.from_polars(self.val_df.select(['input_ids', 'attention_mask', 'label']).collect())
            else:
                self.train_ds, self.val_ds = random_split(self.train_ds, [0.8, 0.2], 
                                                        generator=torch.Generator().manual_seed(123))
                
        if stage == "test":
            self.test_ds = Dataset.from_polars(self.test_df.select(['input_ids', 'attention_mask', 'label']).collect())
            
        if stage == "predict":
            if self.predict_df is not None:
                self.predict_ds = Dataset.from_polars(self.predict_df.select(['input_ids', 'attention_mask', 'label']).collect())
            else:
                self.predict_ds = Dataset.from_polars(self.test_df.select(['input_ids', 'attention_mask', 'label']).collect())

        
    def train_dataloader(self):
        return DataLoader(self.train_ds, collate_fn=self.collator, batch_size=self.batch_size, shuffle=True,
                            num_workers = self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, collate_fn=self.collator, batch_size=self.eval_batch_size,
                        num_workers = self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, collate_fn=self.collator, batch_size=self.eval_batch_size,
                        num_workers = self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict_ds, collate_fn=self.collator, batch_size=self.predict_batch_size,
                        num_workers = self.num_workers)





