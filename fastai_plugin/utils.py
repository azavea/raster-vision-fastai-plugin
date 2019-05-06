import csv

from fastai.callbacks import CSVLogger, Callback
from fastai.metrics import add_metrics
from fastai.torch_core import dataclass, torch, Tensor, Optional, warn

from rastervision.utils.files import (sync_to_dir)


class SyncCallback(Callback):
    def __init__(self, from_dir, to_uri, sync_interval=1):
        self.from_dir = from_dir
        self.to_uri = to_uri
        self.sync_interval = sync_interval

    def on_epoch_end(self, **kwargs):
        if (kwargs['epoch'] + 1) % self.sync_interval == 0:
            sync_to_dir(self.from_dir, self.to_uri)


class ExportCallback(Callback):
    def __init__(self, learn, model_path):
        self.learn = learn
        self.model_path = model_path

    def on_epoch_end(self, **kwargs):
        self.learn.export(self.model_path)


class MyCSVLogger(CSVLogger):
    """Custom CSVLogger

    Modified to:
    - flush after each epoch
    - append to log if already exists
    - use start_epoch
    """
    def __init__(self, learn, filename='history'):
        super().__init__(learn, filename)

    def on_train_begin(self, **kwargs):
        if self.path.exists():
            self.file = self.path.open('a')
        else:
            super().on_train_begin(**kwargs)

    def on_epoch_end(self, epoch, smooth_loss, last_metrics, **kwargs):
        out = super().on_epoch_end(
            epoch, smooth_loss, last_metrics, **kwargs)
        self.file.flush()
        return out


def get_last_epoch(log_path):
    with open(log_path, 'r') as f:
        num_rows = 0
        for row in csv.reader(f):
            num_rows += 1
        if num_rows >= 2:
            return int(row[0])
        return 0


@dataclass
class ConfusionMatrix(Callback):
    "Computes the confusion matrix."
    clas_idx:int=-1

    def on_train_begin(self, **kwargs):
        self.n_classes = 0

    def on_epoch_begin(self, **kwargs):
        self.cm = None

    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
        preds = last_output.argmax(self.clas_idx).view(-1).cpu()
        targs = last_target.view(-1).cpu()
        if self.n_classes == 0:
            self.n_classes = last_output.shape[self.clas_idx]
            self.x = torch.arange(0, self.n_classes)
        cm = ((preds==self.x[:, None]) & (targs==self.x[:, None, None])).sum(dim=2, dtype=torch.float32)
        if self.cm is None: self.cm =  cm
        else:               self.cm += cm

    def on_epoch_end(self, **kwargs):
        self.metric = self.cm

@dataclass
class CMScores(ConfusionMatrix):
    "Base class for metrics which rely on the calculation of the precision and/or recall score."
    average:Optional[str]="binary"      # `binary`, `micro`, `macro`, `weighted` or None
    pos_label:int=1                     # 0 or 1
    eps:float=1e-9
    ignore_idx:int=None

    def _recall(self):
        rec = torch.diag(self.cm) / self.cm.sum(dim=1)
        rec[rec != rec] = 0  # removing potential "nan"s
        if self.average is None: return rec
        else:
            if self.average == "micro": weights = self._weights(avg="weighted")
            else: weights = self._weights(avg=self.average)
            return (rec * weights).sum()

    def _precision(self):
        prec = torch.diag(self.cm) / self.cm.sum(dim=0)
        prec[prec != prec] = 0  # removing potential "nan"s
        if self.average is None: return prec
        else:
            weights = self._weights(avg=self.average)
            return (prec * weights).sum()

    def _weights(self, avg:str):
        if self.n_classes != 2 and avg == "binary":
            avg = self.average = "macro"
            warn("average=`binary` was selected for a non binary case. Value for average has now been set to `macro` instead.")
        if avg == "binary":
            if self.pos_label not in (0, 1):
                self.pos_label = 1
                warn("Invalid value for pos_label. It has now been set to 1.")
            if self.pos_label == 1: return Tensor([0,1])
            else: return Tensor([1,0])
        else:
            if avg == "micro": weights = self.cm.sum(dim=0) / self.cm.sum()
            if avg == "macro": weights = torch.ones((self.n_classes,)) / self.n_classes
            if avg == "weighted": weights = self.cm.sum(dim=1) / self.cm.sum()
            if self.ignore_idx is not None and avg in ["macro", "weighted"]:
                weights[self.ignore_idx] = 0
                weights /= weights.sum()
            return weights

class Recall(CMScores):
    "Compute the Recall."
    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, self._recall())

class Precision(CMScores):
    "Compute the Precision."
    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, self._precision())

@dataclass
class FBeta(CMScores):
    "Compute the F`beta` score."
    beta:float=2

    def on_train_begin(self, **kwargs):
        self.n_classes = 0
        self.beta2 = self.beta ** 2
        self.avg = self.average
        if self.average != "micro": self.average = None

    def on_epoch_end(self, last_metrics, **kwargs):
        prec = self._precision()
        rec = self._recall()
        metric = (1 + self.beta2) * prec * rec / (prec * self.beta2 + rec + self.eps)
        metric[metric != metric] = 0  # removing potential "nan"s
        if self.avg: metric = (self._weights(avg=self.avg) * metric).sum()
        return add_metrics(last_metrics, metric)

    def on_train_end(self, **kwargs): self.average = self.avg