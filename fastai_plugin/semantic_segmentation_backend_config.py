from os.path import join
from copy import deepcopy
from google.protobuf import struct_pb2

import rastervision as rv
from rastervision.backend import (BackendConfig, BackendConfigBuilder)
from rastervision.protos.backend_pb2 import BackendConfig as BackendConfigMsg

from fastai_plugin.semantic_segmentation_backend import SemanticSegmentationBackend

FASTAI_SEMANTIC_SEGMENTATION = 'FASTAI_SEMANTIC_SEGMENTATION'


class SemanticSegmentationBackendConfig(BackendConfig):
    def __init__(self, config):
        super().__init__(FASTAI_SEMANTIC_SEGMENTATION)
        self.config = config

    def create_backend(self, task_config):
        return SemanticSegmentationBackend(self.config, task_config)

    def to_proto(self):
        msg = BackendConfigMsg(
            backend_type=self.backend_type,
            custom_config=struct_pb2.Struct(self.config))
        return msg

    def update_for_command(self, command_type, experiment_config,
                           context=None):
        super().update_for_command(command_type, experiment_config, context)

        if command_type == rv.CHIP:
            self.config['chip_uri'] = experiment_config.chip_uri

        if command_type == rv.TRAIN:
            self.config['train_uri'] = experiment_config.train_uri
            self.config['model_uri'] = join(experiment_config.train_uri, 'model')

    def report_io(self, command_type, io_def):
        super().report_io(command_type, io_def)

        if command_type == rv.CHIP:
            io_def.add_output(self.config['chip_uri'])

        if command_type == rv.TRAIN:
            io_def.add_input(self.config['chip_uri'])
            io_def.add_output(self.config['model_uri'])

    def save_bundle_files(self, bundle_dir):
        if not self.model_uri:
            raise rv.ConfigError('model_uri is not set.')
        local_path, base_name = self.bundle_file(self.model_uri, bundle_dir)
        new_config = self.to_builder() \
                         .with_model_uri(base_name) \
                         .build()
        return (new_config, [local_path])

    def load_bundle_files(self, bundle_dir):
        if not self.model_uri:
            raise rv.ConfigError('model_uri is not set.')
        local_model_uri = join(bundle_dir, self.model_uri)
        return self.to_builder() \
                   .with_model_uri(local_model_uri) \
                   .build()


class SemanticSegmentationBackendConfigBuilder(BackendConfigBuilder):
    def __init__(self, prev_config=None):
        config = {}
        if prev_config:
            config = prev_config.config,
        super().__init__(FASTAI_SEMANTIC_SEGMENTATION,
                         SemanticSegmentationBackendConfig, config, prev_config)
        self.require_task = prev_config is None

    def from_proto(self, msg):
        b = super().from_proto(msg)
        config = msg.custom_config
        b.require_task = None
        return b.with_config(config)

    def validate(self):
        super().validate()

        if not isinstance(self.config, dict):
            raise rv.ConfigError(
                'config must be of type dict, got {}'.format(
                    type(self.config.get('config'))))

        if self.require_task and not self.task:
            raise rv.ConfigError('You must specify the task this backend '
                                 "is for - use 'with_task'.")

        if self.require_task and not isinstance(self.task,
                                                SemanticSegmentationBackendConfig):
            raise rv.ConfigError('Task set with with_task must be of type'
                                 ' SemanticSegmentationConfig, got {}.'.format(
                                     type(self.task)))
        return True

    def build(self):
        self.validate()
        return SemanticSegmentationBackendConfig(deepcopy(self.config))

    def _applicable_tasks(self):
        return [rv.SEMANTIC_SEGMENTATION]

    def _process_task(self):
        pass

    def _load_model_defaults(self, model_defaults):
        pass

    def with_config(self, config):
        b = deepcopy(self)
        b.config.update(config)
        return self

    def with_model_uri(self, model_uri):
        b = deepcopy(self)
        b.config['model_uri'] = model_uri
        return b


def register_plugin(plugin_registry):
    plugin_registry.register_config_builder(
        rv.BACKEND, FASTAI_SEMANTIC_SEGMENTATION,
        SemanticSegmentationBackendConfigBuilder)