from copy import deepcopy

import rastervision as rv

from fastai_plugin.semantic_segmentation_backend import (
    SemanticSegmentationBackend)
from fastai_plugin.simple_backend_config import (
    SimpleBackendConfig, SimpleBackendConfigBuilder)

FASTAI_SEMANTIC_SEGMENTATION = 'FASTAI_SEMANTIC_SEGMENTATION'


class TrainOptions():
    def __init__(self, batch_sz=None, weight_decay=None, lr=None,
                 one_cycle=None,
                 num_epochs=None, model_arch=None, fp16=None,
                 sync_interval=None, debug=None, train_proportion=None):
        self.batch_sz = batch_sz
        self.weight_decay = weight_decay
        self.lr = lr
        self.one_cycle = one_cycle
        self.num_epochs = num_epochs
        self.model_arch = model_arch
        self.fp16 = fp16
        self.sync_interval = sync_interval
        self.debug = debug
        self.train_proportion = train_proportion

    def __setattr__(self, name, value):
        if name in ['batch_sz', 'num_epochs', 'sync_interval']:
            value = int(value) if isinstance(value, float) else value
        super().__setattr__(name, value)


class SemanticSegmentationBackendConfig(SimpleBackendConfig):
    train_opts_class = TrainOptions
    backend_type = FASTAI_SEMANTIC_SEGMENTATION
    backend_class = SemanticSegmentationBackend


class SemanticSegmentationBackendConfigBuilder(SimpleBackendConfigBuilder):
    config_class = SemanticSegmentationBackendConfig

    def _applicable_tasks(self):
        return [rv.SEMANTIC_SEGMENTATION]

    def with_train_options(
            self,
            batch_sz=8,
            weight_decay=1e-2,
            lr=1e-4,
            one_cycle=False,
            num_epochs=5,
            model_arch='resnet18',
            fp16=False,
            sync_interval=1,
            debug=False,
            train_proportion=1.0):
        b = deepcopy(self)
        b.train_opts = TrainOptions(
            batch_sz=batch_sz, weight_decay=weight_decay, lr=lr,
            one_cycle=one_cycle,
            num_epochs=num_epochs, model_arch=model_arch, fp16=fp16,
            sync_interval=sync_interval, debug=debug, train_proportion=train_proportion)
        return b

    def with_pretrained_uri(self, pretrained_uri):
        """pretrained_uri should be uri of exported model file."""
        return super().with_pretrained_uri(pretrained_uri)


def register_plugin(plugin_registry):
    plugin_registry.register_config_builder(
        rv.BACKEND, FASTAI_SEMANTIC_SEGMENTATION,
        SemanticSegmentationBackendConfigBuilder)