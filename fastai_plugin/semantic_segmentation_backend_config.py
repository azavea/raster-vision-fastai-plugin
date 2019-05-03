from os.path import join
from copy import deepcopy

from google.protobuf import struct_pb2

import rastervision as rv
from rastervision.backend import (BackendConfig, BackendConfigBuilder)
from rastervision.protos.backend_pb2 import BackendConfig as BackendConfigMsg
from rastervision.task import SemanticSegmentationConfig

from fastai_plugin.semantic_segmentation_backend import SemanticSegmentationBackend

FASTAI_SEMANTIC_SEGMENTATION = 'FASTAI_SEMANTIC_SEGMENTATION'


class BackendOptions():
    def __init__(self, chip_uri=None, train_uri=None, model_uri=None):
        self.chip_uri = chip_uri
        self.train_uri = train_uri
        self.model_uri = model_uri


class TrainOptions():
    def __init__(self, batch_sz=None, weight_decay=None, lr=None,
                 num_epochs=None, model_arch=None, fp16=None,
                 sync_interval=None, debug=None):
        self.batch_sz = batch_sz
        self.weight_decay = weight_decay
        self.lr = lr
        self.num_epochs = num_epochs
        self.model_arch = model_arch
        self.fp16 = fp16
        self.sync_interval = sync_interval
        self.debug = debug

    def __setattr__(self, name, value):
        if name in ['batch_sz', 'num_epochs', 'sync_interval']:
            value = int(value) if isinstance(value, float) else value
        super().__setattr__(name, value)


class SemanticSegmentationBackendConfig(BackendConfig):
    def __init__(self, backend_opts, train_opts):
        super().__init__(FASTAI_SEMANTIC_SEGMENTATION)
        self.backend_opts = backend_opts
        self.train_opts = train_opts

    def create_backend(self, task_config):
        return SemanticSegmentationBackend(
            task_config, self.backend_opts, self.train_opts)

    def to_proto(self):
        custom_config = struct_pb2.Struct()
        for k, v in self.backend_opts.__dict__.items():
            custom_config[k] = v
        for k, v in self.train_opts.__dict__.items():
            custom_config[k] = v

        msg = BackendConfigMsg(
            backend_type=self.backend_type,
            custom_config=custom_config)
        return msg

    def update_for_command(self, command_type, experiment_config,
                           context=None):
        super().update_for_command(command_type, experiment_config, context)

        if command_type == rv.CHIP:
            self.backend_opts.chip_uri = join(
                experiment_config.chip_uri, 'chips')
        elif command_type == rv.TRAIN:
            self.backend_opts.train_uri = experiment_config.train_uri
            self.backend_opts.model_uri = join(
                experiment_config.train_uri, 'model')

    def report_io(self, command_type, io_def):
        super().report_io(command_type, io_def)

        if command_type == rv.CHIP:
            io_def.add_output(self.backend_opts.chip_uri)
        elif command_type == rv.TRAIN:
            io_def.add_input(self.backend_opts.chip_uri)
            io_def.add_output(self.backend_opts.model_uri)
        elif command_type in [rv.PREDICT, rv.BUNDLE]:
            if not self.backend_opts.model_uri:
                io_def.add_missing('Missing model_uri.')
            else:
                io_def.add_input(self.backend_opts.model_uri)

    def save_bundle_files(self, bundle_dir):
        model_uri = self.backend_opts.model_uri
        if not model_uri:
            raise rv.ConfigError('model_uri is not set.')
        local_path, base_name = self.bundle_file(model_uri, bundle_dir)
        new_config = self.to_builder() \
                         .with_model_uri(base_name) \
                         .build()
        return (new_config, [local_path])

    def load_bundle_files(self, bundle_dir):
        model_uri = self.backend_opts.model_uri
        if not model_uri:
            raise rv.ConfigError('model_uri is not set.')
        local_model_uri = join(bundle_dir, model_uri)
        return self.to_builder() \
                   .with_model_uri(local_model_uri) \
                   .build()


class SemanticSegmentationBackendConfigBuilder(BackendConfigBuilder):
    def __init__(self, prev_config=None):
        self.backend_opts = BackendOptions()
        self.train_opts = TrainOptions()

        if prev_config:
            self.backend_opts = prev_config.backend_opts
            self.train_opts = prev_config.train_opts

        super().__init__(FASTAI_SEMANTIC_SEGMENTATION,
                         SemanticSegmentationBackendConfig)
        self.require_task = prev_config is None

    def from_proto(self, msg):
        b = super().from_proto(msg)
        custom_config = msg.custom_config
        for k in self.backend_opts.__dict__.keys():
            if k in custom_config:
                setattr(b.backend_opts, k, custom_config[k])
        for k in self.train_opts.__dict__.keys():
            if k in custom_config:
                setattr(b.train_opts, k, custom_config[k])
        b.require_task = None
        return b

    def validate(self):
        super().validate()

        if self.require_task and not self.task:
            raise rv.ConfigError('You must specify the task this backend '
                                 "is for - use 'with_task'.")

        if self.require_task and not isinstance(self.task,
                                                SemanticSegmentationConfig):
            raise rv.ConfigError('Task set with with_task must be of type'
                                 ' SemanticSegmentationConfig, got {}.'.format(
                                     type(self.task)))
        return True

    def build(self):
        self.validate()
        return SemanticSegmentationBackendConfig(
            self.backend_opts, self.train_opts)

    def _applicable_tasks(self):
        return [rv.SEMANTIC_SEGMENTATION]

    def _process_task(self):
        return self

    def with_model_uri(self, model_uri):
        b = deepcopy(self)
        b.backend_opts.model_uri = model_uri
        return b

    def with_train_options(
            self,
            batch_sz=8,
            weight_decay=1e-2,
            lr=1e-4,
            num_epochs=5,
            model_arch='resnet18',
            fp16=False,
            sync_interval=1,
            debug=False):
        b = deepcopy(self)
        b.train_opts = TrainOptions(
            batch_sz=batch_sz, weight_decay=weight_decay, lr=lr,
            num_epochs=num_epochs, model_arch=model_arch, fp16=fp16,
            sync_interval=sync_interval, debug=debug)
        return b


def register_plugin(plugin_registry):
    plugin_registry.register_config_builder(
        rv.BACKEND, FASTAI_SEMANTIC_SEGMENTATION,
        SemanticSegmentationBackendConfigBuilder)