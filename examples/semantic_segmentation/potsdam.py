import os
from os.path import join

import rastervision as rv
from examples.utils import str_to_bool, save_image_crop

from fastai_plugin.semantic_segmentation_backend_config import (
    FASTAI_SEMANTIC_SEGMENTATION)


class PotsdamSemanticSegmentation(rv.ExperimentSet):
    def get_exp(self, exp_id, config, raw_uri, processed_uri, root_uri,
                pred_chip_size=300, test=False):
        """Run an experiment on the ISPRS Potsdam dataset.

        Args:
            raw_uri: (str) directory of raw data
            root_uri: (str) root directory for experiment output
            test: (bool) if True, run a very small experiment as a test and generate
                debug output
        """
        test = str_to_bool(test)
        train_ids = ['2-10', '2-11', '3-10', '3-11', '4-10', '4-11', '4-12', '5-10',
                     '5-11', '5-12', '6-10', '6-11', '6-7', '6-9', '7-10', '7-11',
                     '7-12', '7-7', '7-8', '7-9']
        val_ids = ['2-12', '3-12', '6-12']
        # infrared, red, green
        channel_order = [3, 0, 1]

        chip_key = 'potsdam-seg'
        if test:
            config['debug'] = True
            config['batch_sz'] = 1
            config['num_epochs'] = 1

            train_ids = train_ids[0:1]
            val_ids = val_ids[0:1]
            exp_id += '-test'
            chip_key += '-test'

        classes = {
            'Car': (1, '#ffff00'),
            'Building': (2, '#0000ff'),
            'Low Vegetation': (3, '#00ffff'),
            'Tree': (4, '#00ff00'),
            'Impervious': (5, "#ffffff"),
            'Clutter': (6, "#ff0000")
        }

        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_chip_size(300) \
                            .with_classes(classes) \
                            .with_chip_options(window_method='sliding',
                                               stride=300, debug_chip_probability=1.0) \
                            .build()

        backend = rv.BackendConfig.builder(FASTAI_SEMANTIC_SEGMENTATION) \
                                  .with_task(task) \
                                  .with_train_options(**config) \
                                  .build()

        def make_scene(id):
            id = id.replace('-', '_')
            raster_uri = '{}/4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'.format(
                raw_uri, id)

            label_uri = '{}/5_Labels_for_participants/top_potsdam_{}_label.tif'.format(
                raw_uri, id)

            if test:
                crop_uri = join(
                    processed_uri, 'crops', os.path.basename(raster_uri))
                save_image_crop(raster_uri, crop_uri, size=600)
                raster_uri = crop_uri

            # Using with_rgb_class_map because label TIFFs have classes encoded as RGB colors.
            label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                .with_rgb_class_map(task.class_map) \
                .with_raster_source(label_uri) \
                .build()

            # URI will be injected by scene config.
            # Using with_rgb(True) because we want prediction TIFFs to be in RGB format.
            label_store = rv.LabelStoreConfig.builder(rv.SEMANTIC_SEGMENTATION_RASTER) \
                .with_rgb(True) \
                .build()

            scene = rv.SceneConfig.builder() \
                                  .with_task(task) \
                                  .with_id(id) \
                                  .with_raster_source(raster_uri,
                                                      channel_order=channel_order) \
                                  .with_label_source(label_source) \
                                  .with_label_store(label_store) \
                                  .build()

            return scene

        train_scenes = [make_scene(id) for id in train_ids]
        val_scenes = [make_scene(id) for id in val_ids]

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes(train_scenes) \
                                  .with_validation_scenes(val_scenes) \
                                  .build()

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id(exp_id) \
                                        .with_chip_key(chip_key) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .with_root_uri(root_uri) \
                                        .build()

        return experiment

    def exp_resnet18(self, raw_uri, processed_uri, root_uri, test=False):
        exp_id = 'resnet18'
        config = {
            'batch_sz': 8,
            'num_epochs': 5,
            'debug': False,
            'lr': 1e-4,
            'sync_interval': 10,
            'model_arch': 'resnet18'
        }
        return self.get_exp(exp_id, config, raw_uri, processed_uri, root_uri,
                            test)

    def exp_resnet18_better(self, raw_uri, processed_uri, root_uri, test=False):
        # A better set of hyperparams.
        exp_id = 'resnet18_better'
        config = {
            'batch_sz': 16,
            'num_epochs': 20,
            'debug': False,
            'lr': 1e-4,
            'one_cycle': True,
            'sync_interval': 1,
            'tta': True,
            'model_arch': 'resnet18',
            'flip_vert': True
        }
        pred_chip_size = 1200
        return self.get_exp(exp_id, config, raw_uri, processed_uri, root_uri,
                            test, pred_chip_size=pred_chip_size)

    def exp_resnet50(self, raw_uri, processed_uri, root_uri, test=False):
        exp_id = 'resnet50'
        config = {
            'batch_sz': 8,
            'num_epochs': 5,
            'debug': False,
            'lr': 1e-4,
            'sync_interval': 10,
            'model_arch': 'resnet50'
        }
        return self.get_exp(exp_id, config, raw_uri, processed_uri, root_uri,
                            test)

    # Example of experiment using half of the training chips
    def exp_subset_half_train_data(self, raw_uri, processed_uri, root_uri, test=False):
        exp_id = 'resnet18-half_train_data'
        config = {
            'batch_sz': 8,
            'num_epochs': 5,
            'debug': False,
            'lr': 1e-4,
            'sync_interval': 10,
            'model_arch': 'resnet18',
            'train_prop': 0.5
        }
        return self.get_exp(exp_id, config, raw_uri, processed_uri, root_uri,
                            test)

    # Example of experiment using exactlty 5,000 chips
    def exp_subset_5k_chips(self, raw_uri, processed_uri, root_uri, test=False):
        exp_id = 'resnet18-5k_train_chips'
        config = {
            'batch_sz': 8,
            'num_epochs': 5,
            'debug': False,
            'lr': 1e-4,
            'sync_interval': 10,
            'model_arch': 'resnet18',
            'train_count': 5000
        }
        return self.get_exp(exp_id, config, raw_uri, processed_uri, root_uri,
                            test)


if __name__ == '__main__':
    rv.main()
