from transformers import EsmForTokenClassification
import lightning as L
import torch
from torchmetrics.functional import accuracy
from torch import optim
import polars as pl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EsmForSecondaryStructure(L.LightningModule):
  def __init__(self,
               num_labels: int = 10,
               pretrained: str = "facebook/esm2_t6_8M_UR50D",
               ckpt_path = None,
               warmup_epochs: int = 0,
               decay_epochs: int = 3,
               learning_rate: float = 5e-5,
               weight_decay: float = 0.0,
               eps: float = 1e-8,
               input_key: str = "input_ids",
               label_key: str = "label",
               mask_key: str = "attention_mask",
               output_key: str = "logits",
               loss_key: str = "loss",
               create_emb_df: bool = False
               ):
    super().__init__()
    self.model = EsmForTokenClassification.from_pretrained(pretrained,
                                                           num_labels=num_labels,
                                                           dtype="auto",
                                                           output_hidden_states = True
                                                           ).train()
    if ckpt_path:
      sd = torch.load(ckpt_path,
                      map_location=device)['state_dict']
      sd = {k[6:] : sd[k] for k in sd.keys()}

      num_pretrained_labels = sd['classifier.bias'].shape[0]
      diff = num_pretrained_labels - num_labels

      if diff > 0:
      # removes values for labels [0, diff - 1]
        sd['classifier.bias'] = sd['classifier.bias'][diff:]
        sd['classifier.weight'] = sd['classifier.weight'][diff:]
      elif diff < 0:
      # repeats value for label [C] * diff
        sd['classifier.bias'] = torch.hstack([sd['classifier.bias'], sd['classifier.bias'][-1].repeat(-diff)])
        sd['classifier.weight'] = torch.vstack([sd['classifier.weight'], sd['classifier.weight'][-1].repeat(-diff, 1)])

      self.model.load_state_dict(sd)

    self.num_labels = num_labels
    self.accuracy = accuracy
    
    self.warmup_epochs = warmup_epochs
    self.decay_epochs = decay_epochs
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay
    self.eps = eps

    self.input_key = input_key
    self.label_key = label_key
    self.mask_key = mask_key
    self.output_key = output_key
    self.loss_key = loss_key

    if create_emb_df:
      self.emb_df =  pl.LazyFrame(schema = {'input_ids' : pl.List(pl.Int64),
                                            'label' : pl.List(pl.Int64),
                                            'attention_mask' : pl.List(pl.Int64),
                                            'embedding' : pl.List(pl.List(pl.Float64))})
    else:
      self.emb_df = None


  def compute_accuracy(self, predictions, labels):
    # if all labels are -100, return acc = None
    if torch.where((labels == -100), 0, labels).sum() == 0:
      acc = None
    else:
      acc = self.accuracy(
          predictions,
          labels,
          num_classes=self.num_labels,
          task="multiclass", 
          ignore_index = -100
      )

    return acc

  def forward(self, batch):
    outputs = self.model(
        batch[self.input_key],
        attention_mask=batch[self.mask_key],
        labels=batch[self.label_key]
    )
    return outputs 

  def training_step(self, batch, batch_idx):
    outputs = self.forward(batch)

    logits = outputs[self.output_key]
    predictions = torch.argmax(logits, 2)
    labels = batch[self.label_key]

    acc = self.compute_accuracy(predictions, labels)
    # if all labels are -100, return loss as 0.0
    # accuracy still works when all labels are -100 but loss returns as NaN 
    if acc:
      self.log("train_loss", outputs[self.loss_key], on_step=False, on_epoch=True, prog_bar=True)
      self.log("train_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)
      return outputs[self.loss_key]
    else:
      return torch.tensor(0.0).requires_grad_()


  def validation_step(self, batch, batch_idx):
    outputs = self.forward(batch)
    
    logits = outputs[self.output_key]
    predictions = torch.argmax(logits, 2)
    labels = batch[self.label_key]

    acc = self.compute_accuracy(predictions, labels)
    if acc:
      self.log("val_loss", outputs[self.loss_key], on_step=False, on_epoch=True, prog_bar=True)
      self.log("val_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)

  def test_step(self, batch, batch_idx):
    outputs = self.forward(batch)

    logits = outputs[self.output_key]
    predictions = torch.argmax(logits, 2)
    labels = batch[self.label_key]

    acc = self.compute_accuracy(predictions, labels)
    if acc:
      self.log("test_loss", outputs[self.loss_key], on_step=False, on_epoch=True, prog_bar=True)
      self.log("test_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)

  def predict(self, batch):
    with torch.no_grad():
      outputs = self.model(
          batch[self.input_key],
          attention_mask=batch[self.mask_key],
      )
      return outputs
      
  def get_hidden_states(self, batch):
    hidden_states = self.predict(batch).hidden_states
    last_layer = hidden_states[-2].squeeze(0) #.numpy()
    embedding = hidden_states[-1].squeeze(0) #.numpy()
    return last_layer, embedding

  def predict_step(self, batch, batch_idx):
    if self.emb_df is not None:
      _, embedding = self.get_hidden_states(batch)
      self.emb_df =  pl.concat([self.emb_df,    
                                pl.LazyFrame([{'input_ids'      : batch['input_ids'][0].tolist(),
                                                'label'          : batch['label'][0].tolist(),
                                                'attention_mask' : batch['attention_mask'][0].tolist(),
                                                'embedding'      : embedding.tolist()}])
                                ])
      return None
    else:
      outputs = self.predict(batch)
      return outputs[self.output_key].argmax(2)
  

  def configure_optimizers(self):
    optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
                            eps=self.eps)

    decay_scheduler = optim.lr_scheduler.LinearLR(optimizer,
                                        start_factor = 1.0,
                                        end_factor = 0.0,
                                        total_iters=self.decay_epochs)

    if self.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer,
                                                start_factor = 0.01,
                                                end_factor = 1.0,
                                                total_iters=self.warmup_epochs)

        scheduler = optim.lr_scheduler.SequentialLR(optimizer,
                                                    schedulers = [warmup_scheduler, decay_scheduler],
                                                    milestones = [self.warmup_epochs])
    else:
      scheduler = decay_scheduler

      return [optimizer], [scheduler]
