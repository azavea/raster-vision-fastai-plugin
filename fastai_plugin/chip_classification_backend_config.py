from copy import deepcopy

import rastervision as rv

from fastai_plugin.chip_classification_backend import (
    ChipClassificationBackend)
from fastai_plugin.simple_backend_config import (
    SimpleBackendConfig, SimpleBackendConfigBuilder)

FASTAI_CHIP_CLASSIFICATION = 'FASTAI_CHIP_CLASSIFICATION'


class TrainOptions():
    def __init__(self, batch_sz=None, weight_decay=None, lr=None,
                 one_cycle=None,
                 num_epochs=None, model_arch=None, fp16=None,
                 flip_vert=None, sync_interval=None, debug=None,
                 train_prop=None, train_count=None):
        self.batch_sz = batch_sz
        self.weight_decay = weight_decay
        self.lr = lr
        self.one_cycle = one_cycle
        self.num_epochs = num_epochs
        self.model_arch = model_arch
        self.fp16 = fp16
        self.flip_vert = flip_vert
        self.sync_interval = sync_interval
        self.debug = debug
        self.train_prop = train_prop
        self.train_count = train_count

    def __setattr__(self, name, value):
        if name in ['batch_sz', 'num_epochs', 'sync_interval']:
            value = int(value) if isinstance(value, float) else value
        super().__setattr__(name, value)


class ChipClassificationBackendConfig(SimpleBackendConfig):
    train_opts_class = TrainOptions
    backend_type = FASTAI_CHIP_CLASSIFICATION
    backend_class = ChipClassificationBackend


class ChipClassificationBackendConfigBuilder(SimpleBackendConfigBuilder):
    config_class = ChipClassificationBackendConfig

    def _applicable_tasks(self):
        return [rv.CHIP_CLASSIFICATION]

    def with_train_options(
            self,
            batch_sz=8,
            weight_decay=1e-2,
            lr=1e-4,
            one_cycle=True,
            num_epochs=1,
            model_arch='resnet18',
            fp16=False,
            flip_vert=True,
            sync_interval=1,
            debug=False,
            train_prop=1.0,
            train_count=None):
        """Set options for training models.

        Args:
            batch_sz: (int) the batch size
            weight_decay: (float) the weight decay
            lr: (float or None) the learning rate if using a fixed LR (ie. one_cycle is False),
                or the maximum LR to use if one_cycle is True,
                or None if automatic learning rate finder (fastai lr_find)
                should be used
            one_cycle: (bool) True if fastai fit_one_cycle should be used. This
                cycles the LR once during the course of training and seems to
                result in a pretty consistent improvement. See lr for more
                details.
            num_epochs: (int) number of epochs (sweeps through training set) to
                train model for
            model_arch: (str) classification model backbone to use for UNet
                architecture. Any option in torchvision.models is valid, for
                example, resnet18.
            fp16: (bool) use mixed-precision training. Ideally, this will make
                things run 2x fast.
            flip_vert: (bool) use vertical flips and rotations for data aug
            sync_interval: (int) sync training directory to cloud every
                sync_interval epochs.
            debug: (bool) if True, save debug chips (ie. visualizations of
                input to model during training) during training and use
                single-core for creating minibatches.
            train_prop: (float) number between 0 and 1 that controls what
                proportion of the training set is used for training
            train_count: (int) number of training examples to use during
                training
        """
        b = deepcopy(self)
        b.train_opts = TrainOptions(
            batch_sz=batch_sz, weight_decay=weight_decay, lr=lr,
            one_cycle=one_cycle,
            num_epochs=num_epochs, model_arch=model_arch, fp16=fp16,
            flip_vert=flip_vert, sync_interval=sync_interval, debug=debug,
            train_prop=train_prop, train_count=train_count)
        return b

    def with_pretrained_uri(self, pretrained_uri):
        """pretrained_uri should be uri of exported model file."""
        return super().with_pretrained_uri(pretrained_uri)


def register_plugin(plugin_registry):
    plugin_registry.register_config_builder(
        rv.BACKEND, FASTAI_CHIP_CLASSIFICATION,
        ChipClassificationBackendConfigBuilder)
