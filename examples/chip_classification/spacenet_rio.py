import os
from os.path import join

import rastervision as rv
from examples.utils import get_scene_info, str_to_bool, save_image_crop

from fastai_plugin.chip_classification_backend_config import (
    FASTAI_CHIP_CLASSIFICATION)

aoi_path = 'AOI_1_Rio/srcData/buildingLabels/Rio_OUTLINE_Public_AOI.geojson'


class ChipClassificationExperiments(rv.ExperimentSet):
    def get_exp(self, exp_id, config, raw_uri, processed_uri, root_uri, test=False):
        """Chip classification experiment on Spacenet Rio dataset.
        Run the data prep notebook before running this experiment. Note all URIs can be
        local or remote.
        Args:
            raw_uri: (str) directory of raw data
            processed_uri: (str) directory of processed data
            root_uri: (str) root directory for experiment output
            test: (bool) if True, run a very small experiment as a test and generate
                debug output
        """
        test = str_to_bool(test)
        train_scene_info = get_scene_info(join(processed_uri, 'train-scenes.csv'))
        val_scene_info = get_scene_info(join(processed_uri, 'val-scenes.csv'))

        chip_key = 'spacenet-rio-chip-classification'
        if test:
            exp_id += '-test'
            config['num_epochs'] = 1
            config['batch_sz'] = 32
            config['debug'] = True
            train_scene_info = train_scene_info[0:4]
            val_scene_info = val_scene_info[0:4]

        task = rv.TaskConfig.builder(rv.CHIP_CLASSIFICATION) \
                            .with_chip_size(200) \
                            .with_classes({
                                'building': (1, 'red'),
                                'no_building': (2, 'black')
                            }) \
                            .build()

        backend = rv.BackendConfig.builder(FASTAI_CHIP_CLASSIFICATION) \
                                  .with_task(task) \
                                  .with_train_options(**config) \
                                  .build()

        def make_scene(scene_info):
            (raster_uri, label_uri) = scene_info
            raster_uri = join(raw_uri, raster_uri)
            label_uri = join(processed_uri, label_uri)
            aoi_uri = join(raw_uri, aoi_path)

            # if test:
                # crop_uri = join(processed_uri, 'crops', os.path.basename(raster_uri))
                # save_image_crop(raster_uri, crop_uri, label_uri=label_uri,
                #                 size=600, min_features=10)
                # raster_uri = crop_uri

            id = os.path.splitext(os.path.basename(raster_uri))[0]
            label_source = rv.LabelSourceConfig.builder(rv.CHIP_CLASSIFICATION) \
                                               .with_uri(label_uri) \
                                               .with_ioa_thresh(0.5) \
                                               .with_use_intersection_over_cell(False) \
                                               .with_pick_min_class_id(True) \
                                               .with_background_class_id(2) \
                                               .with_infer_cells(True) \
                                               .build()

            return rv.SceneConfig.builder() \
                                 .with_task(task) \
                                 .with_id(id) \
                                 .with_raster_source(raster_uri) \
                                 .with_label_source(label_source) \
                                 .with_aoi_uri(aoi_uri) \
                                 .build()

        train_scenes = [make_scene(info) for info in train_scene_info]
        val_scenes = [make_scene(info) for info in val_scene_info]

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes(train_scenes) \
                                  .with_validation_scenes(val_scenes) \
                                  .build()

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id(exp_id) \
                                        .with_chip_key(chip_key) \
                                        .with_root_uri(root_uri) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
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
            'model_arch': 'resnet18',
        }
        return self.get_exp(exp_id, config, raw_uri, processed_uri, root_uri, test)

    def exp_resnet50(self, raw_uri, processed_uri, root_uri, test=False):
        exp_id = 'resnet50'
        config = {
            'batch_sz': 8,
            'num_epochs': 5,
            'debug': False,
            'lr': 1e-4,
            'sync_interval': 10,
            'model_arch': 'resnet50',
        }
        return self.get_exp(exp_id, config, raw_uri, processed_uri, root_uri, test)


if __name__ == '__main__':
    rv.main()
