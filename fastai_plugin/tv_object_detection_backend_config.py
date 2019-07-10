from copy import deepcopy

import rastervision as rv

from fastai_plugin.tv_object_detection_backend import (
    ObjectDetectionBackend)
from fastai_plugin.simple_backend_config import (
    SimpleBackendConfig, SimpleBackendConfigBuilder)

TV_OBJECT_DETECTION = 'TV_OBJECT_DETECTION'


class TrainOptions():
    def __init__(self, batch_sz=None, lr=None,
                 num_epochs=None, model_arch=None,
                 sync_interval=None, debug=None):
        self.batch_sz = batch_sz
        self.lr = lr
        self.num_epochs = num_epochs
        self.model_arch = model_arch
        self.sync_interval = sync_interval
        self.debug = debug

    def __setattr__(self, name, value):
        if name in ['batch_sz', 'num_epochs', 'sync_interval']:
            value = int(value) if isinstance(value, float) else value
        super().__setattr__(name, value)


class ObjectDetectionBackendConfig(SimpleBackendConfig):
    train_opts_class = TrainOptions
    backend_type = TV_OBJECT_DETECTION
    backend_class = ObjectDetectionBackend


class ObjectDetectionBackendConfigBuilder(SimpleBackendConfigBuilder):
    config_class = ObjectDetectionBackendConfig

    def _applicable_tasks(self):
        return [rv.OBJECT_DETECTION]

    def with_train_options(
            self,
            batch_sz=8,
            lr=1e-4,
            num_epochs=5,
            model_arch='maskrcnn_resnet50_fpn',
            sync_interval=1,
            debug=False):
        b = deepcopy(self)
        b.train_opts = TrainOptions(
            batch_sz=batch_sz, lr=lr,
            num_epochs=num_epochs, model_arch=model_arch,
            sync_interval=sync_interval, debug=debug)
        return b

    def with_pretrained_uri(self, pretrained_uri):
        """pretrained_uri should be uri of exported model file."""
        return super().with_pretrained_uri(pretrained_uri)


def register_plugin(plugin_registry):
    plugin_registry.register_config_builder(
        rv.BACKEND, TV_OBJECT_DETECTION,
        ObjectDetectionBackendConfigBuilder)