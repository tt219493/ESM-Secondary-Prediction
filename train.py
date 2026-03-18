import collections
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities import rank_zero_only
import matplotlib.pyplot as plt

class CustomLogger(Logger):
    def __init__(self):
      super().__init__()

      self.history = collections.defaultdict(list)

    @property
    def name(self):
      return "Logger"

    @property
    def version(self):
      return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
      # params is an argparse.Namespace
      pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
      for metric_name, metric_value in metrics.items():
        if metric_name != 'epoch':
            self.history[metric_name].append(metric_value)
        else:
            if (not len(self.history['epoch']) or
                not self.history['epoch'][-1] == metric_value) :
                self.history['epoch'].append(metric_value)
            else:
                pass
      return
    

def plot_training(logger, dimensions = (15, 5)):
    x_axis = logger.history['epoch']
    train_loss = logger.history['train_loss']
    val_loss = logger.history['val_loss']
    train_acc = logger.history['train_acc']
    val_acc = logger.history['val_acc']

    plt.figure(figsize=dimensions)
    plt.subplot(1, 2, 1)
    plt.plot(x_axis, train_loss, label='train')
    plt.plot(x_axis, val_loss, label='validation')
    plt.legend()
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(x_axis, train_acc, label='train')
    plt.plot(x_axis, val_acc, label='validation')
    plt.legend()
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.title("Accuracy")
    plt.show()
   