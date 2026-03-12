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
               max_length: int = 1022,
               pretrained: str = "facebook/esm2_t6_8M_UR50D",
               ckpt_path = None,
               warmup_epochs: int = 0,
               decay_epochs: int = 3,
               learning_rate: float = 5e-5,
               weight_decay: float = 0.0,
               eps: float = 1e-8,
               start_factor: float = 0.01,
               end_factor: float = 1.0,
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
    self.max_length = max_length
    self.accuracy = accuracy

    self.warmup_epochs = warmup_epochs
    self.decay_epochs = decay_epochs
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay
    self.eps = eps
    self.start_factor = start_factor
    self.end_factor = end_factor

    self.input_key = input_key
    self.label_key = label_key
    self.mask_key = mask_key
    self.output_key = output_key
    self.loss_key = loss_key

    self.create_emb_df = create_emb_df

  def compute_accuracy(self, predictions, labels):
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
    total_res = torch.where(batch['label'] != -100, 1, 0).sum().to(device)
    if total_res == 0:
        return torch.tensor(0.0).requires_grad_()

    if batch[self.input_key].shape[1] > self.max_length:
        id_split = batch[self.input_key].split(self.max_length, dim=1)
        mask_split = batch[self.mask_key].split(self.max_length, dim=1)
        label_split = batch[self.label_key].split(self.max_length, dim=1)

        logits = []
        loss = torch.tensor(0.0).to(device)

        for i, m, l  in zip(id_split, mask_split, label_split):
            outputs = self.model(i.contiguous(), attention_mask=m.contiguous(), labels=l.contiguous())
            if torch.isnan(outputs[self.loss_key]).any():
              loss += torch.tensor(0.0).requires_grad_()
            else:
              loss += outputs[self.loss_key] * torch.where(l != -100, 1, 0).sum()
            logits.append(outputs[self.output_key])

        loss /= total_res
        logits = torch.cat(logits, axis=1)
    else:
        outputs = self.forward(batch)
        logits = outputs[self.output_key]
        loss = outputs[self.loss_key]

    predictions = torch.argmax(logits, 2)
    labels = batch[self.label_key]

    acc = self.compute_accuracy(predictions, labels)
    self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    return loss


  def validation_step(self, batch, batch_idx):
    total_res = torch.where(batch['label'] != -100, 1, 0).sum().to(device)
    if total_res == 0:
        pass

    if batch[self.input_key].shape[1] > self.max_length:
        id_split = batch[self.input_key].split(self.max_length, dim=1)
        mask_split = batch[self.mask_key].split(self.max_length, dim=1)
        label_split = batch[self.label_key].split(self.max_length, dim=1)

        logits = []
        loss = torch.tensor(0.0).to(device)

        for i, m, l  in zip(id_split, mask_split, label_split):
            outputs = self.model(i.contiguous(), attention_mask=m.contiguous(), labels=l.contiguous())
            if torch.isnan(outputs[self.loss_key]).any():
              loss += torch.tensor(0.0).requires_grad_()
            else:
              loss += outputs[self.loss_key] * torch.where(l != -100, 1, 0).sum()
            logits.append(outputs[self.output_key])

        loss /= total_res
        logits = torch.cat(logits, axis=1)
    else:
        outputs = self.forward(batch)
        logits = outputs[self.output_key]
        loss = outputs[self.loss_key]

    predictions = torch.argmax(logits, 2)
    labels = batch[self.label_key]

    acc = self.compute_accuracy(predictions, labels)
    self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

  def test_step(self, batch, batch_idx):
    total_res = torch.where(batch['label'] != -100, 1, 0).sum().to(device)
    if total_res == 0:
        pass

    if batch[self.input_key].shape[1] > self.max_length:
        id_split = batch[self.input_key].split(self.max_length, dim=1)
        mask_split = batch[self.mask_key].split(self.max_length, dim=1)
        label_split = batch[self.label_key].split(self.max_length, dim=1)

        logits = []
        loss = torch.tensor(0.0).to(device)

        for i, m, l  in zip(id_split, mask_split, label_split):
            outputs = self.model(i.contiguous(), attention_mask=m.contiguous(), labels=l.contiguous())
            if torch.isnan(outputs[self.loss_key]).any():
              loss += torch.tensor(0.0).requires_grad_()
            else:
              loss += outputs[self.loss_key] * torch.where(l != -100, 1, 0).sum()
            logits.append(outputs[self.output_key])

        loss /= total_res
        logits = torch.cat(logits, axis=1)
    else:
        outputs = self.forward(batch)
        logits = outputs[self.output_key]
        loss = outputs[self.loss_key]

    predictions = torch.argmax(logits, 2)
    labels = batch[self.label_key]

    acc = self.compute_accuracy(predictions, labels)
    self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

  def predict(self, batch):
    with torch.no_grad():
      outputs = self.model(
          batch[self.input_key],
          attention_mask=batch[self.mask_key]
      )
      return outputs

  def predict_step(self, batch, batch_idx):
    if batch[self.input_key].shape[1] > self.max_length:
        id_split = batch[self.input_key].split(self.max_length, dim=1)
        mask_split = batch[self.mask_key].split(self.max_length, dim=1)

        logits = []
        embedding = []

        for i, m  in zip(id_split, mask_split):
            with torch.no_grad():
                outputs = self.model(i.contiguous(), attention_mask=m.contiguous())

            logits.append(outputs[self.output_key])
            embedding.append(outputs['hidden_states'][-1])

        logits = torch.cat(logits, axis=1)
        embedding = torch.cat(embedding, axis=1)
    else:
        outputs = self.predict(batch)
        logits = outputs[self.output_key]
        embedding = outputs['hidden_states'][-1]


    # included here just so progress bar can be seen when creating creating embedding df
    # currently used to create df for input into other models
    if self.create_emb_df:
      return pl.LazyFrame([{'input_ids'      : batch['input_ids'].tolist(),
                            'label'          : batch['label'].tolist(),
                            'attention_mask' : batch['attention_mask'].tolist(),
                            'embedding'      : embedding.tolist()}])
    else:
      return logits.argmax(2)

  def get_embedding(self, batch):
    # use to directly feed embeddings to other models
    if batch[self.input_key].shape[1] > self.max_length:
        id_split = batch[self.input_key].split(self.max_length, dim=1)
        mask_split = batch[self.mask_key].split(self.max_length, dim=1)

        embedding = []

        for i, m  in zip(id_split, mask_split):
            with torch.no_grad():
                outputs = self.model(i, attention_mask=m)
            embedding.append(outputs['hidden_states'][-1])

        embedding = torch.cat(embedding, axis=1)
        return {
        'input_ids' : batch[self.input_key],
        'label' : batch[self.label_key],
        'embedding': embedding
        }
    else:
        outputs = self.predict(batch)
        return {
        'input_ids' : batch[self.input_key],
        'label' : batch[self.label_key],
        'embedding' : outputs['hidden_states'][-1]
        }

  def configure_optimizers(self):
    optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
                            eps=self.eps)

    decay_scheduler = optim.lr_scheduler.LinearLR(optimizer,
                                        start_factor = 1.0,
                                        end_factor = self.end_factor,
                                        total_iters=self.decay_epochs)

    if self.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer,
                                                start_factor = self.start_factor,
                                                end_factor = 1.0,
                                                total_iters=self.warmup_epochs)

        scheduler = optim.lr_scheduler.SequentialLR(optimizer,
                                                    schedulers = [warmup_scheduler, decay_scheduler],
                                                    milestones = [self.warmup_epochs])
    else:
      scheduler = decay_scheduler

      return [optimizer], [scheduler]
