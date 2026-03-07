from transformers import EsmForTokenClassification
import lightning as L
import torch
from torchmetrics.functional import accuracy
from torch import optim

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

  def compute_accuracy(self, predictions, labels):
    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        true_predictions = true_predictions + [
            p for (p, l) in zip(prediction, label) if l != -100
        ]
        true_labels = true_labels + [
            l for l in label if l != -100
        ]
        
    if len(true_labels) == 0:
      acc = None
    else:
      acc = self.accuracy(
          torch.tensor(true_predictions),
          torch.tensor(true_labels),
          num_classes=self.num_labels,
          task="multiclass", 
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
          torch.tensor(batch[self.input_key]).reshape(1, -1).to(device),
          attention_mask=torch.tensor(batch[self.mask_key]).reshape(1, -1).to(device),
      )
      return outputs

  def predict_step(self, batch, batch_idx):
    outputs = self.predict(batch)
    logits = outputs[self.output_key]
    predictions = torch.argmax(logits, 2)

    return predictions
  
  def get_hidden_states(self, batch):
    hidden_states = self.predict(batch).hidden_states
    last_layer = hidden_states[-2].squeeze(0).numpy()
    embeddings = hidden_states[-1].squeeze(0).numpy()
    return last_layer, embeddings


#  {
#       'id' : batch['id'][0],
#       'asym_id' : batch['asym_id'][0],
#       'sequence' : batch['sequence'][0],
#       'index' : torch.tensor(batch['index']).tolist(),
#       'label' : torch.tensor(batch['label']).tolist(),
#       'prediction' : predictions.flatten().tolist()
#    }


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
