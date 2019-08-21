import os
from os.path import join

import rastervision as rv
from examples.utils import str_to_bool, save_image_crop

from fastai_plugin.chip_classification_backend_config import (
    FASTAI_CHIP_CLASSIFICATION)



class CowcChipClassificationZXYExperiments(rv.ExperimentSet):
    def exp_main(self, raw_uri, processed_uri, root_uri, test=False):
        """Chip Classification on COWC (Cars Overhead with Context) Potsdam

        Args:
            raw_uri: (str) directory of raw data
            processed_uri: (str) directory of processed data
            root_uri: (str) root directory for experiment output
            test: (bool) if True, run a very small experiment as a test and generate
                debug output
        """
        test = str_to_bool(test)
        exp_id = 'cowc-chip-classification-zxy'
        num_epochs = 10
        batch_sz = 64
        debug = False
        lr = 2e-4
        model_arch = 'resnet50'
        mixed_prec = True
        sync_interval = 10
        train_scene_ids = ['2_10', '2_11', '2_12', '2_14', '3_11',
                           '3_13', '4_10', '5_10', '6_7', '6_9']
        val_scene_ids = ['2_13', '6_8', '3_10']

        if test:
            exp_id += '-test'
            num_epochs = 10
            batch_sz = 2
            debug = True
            train_scene_ids = train_scene_ids[0:1]
            val_scene_ids = val_scene_ids[0:1]

        task = rv.TaskConfig.builder(rv.CHIP_CLASSIFICATION) \
            .with_chip_size(256) \
            .with_classes({
                'car': (1, 'red'),
                'background': (2, 'black')
            }) \
            .build()

        config = {
            'batch_sz': batch_sz,
            'num_epochs': num_epochs,
            'debug': debug,
            'lr': lr,
            'sync_interval': sync_interval,
            'model_arch': model_arch,
            'mixed_prec': mixed_prec,
            'log_tensorboard': True,
            'run_tensorboard': True
        }

        backend = rv.BackendConfig.builder(FASTAI_CHIP_CLASSIFICATION) \
                        .with_task(task) \
                        .with_train_options(**config) \
                        .build()

        def make_scene(id):
            label_uri = join(
                processed_uri, 'labels', 'all', 'top_potsdam_{}_RGBIR.json'.format(id))

            # bounds = [min_lat, min_lng, max_lat, max_lng]
            # TODO
            bounds = [0, 0, 1, 1]
            raster_source = rv.RasterSourceConfig.builder(rv.ZXY_RASTER_SOURCE) \
                .with_tile_schema('https://tiles.rasterfoundry.com/10f0a913-4ef1-4222-875b-4b2b8fc5963a/{z}/{x}/{y}/?tag=1566233879794') \
                .with_zoom(21) \
                .with_bounds(bounds) \
                .build()

            label_source = rv.LabelSourceConfig.builder(rv.CHIP_CLASSIFICATION) \
                .with_uri(label_uri) \
                .with_ioa_thresh(0.5) \
                .with_use_intersection_over_cell(False) \
                .with_pick_min_class_id(True) \
                .with_background_class_id(2) \
                .with_infer_cells(True) \
                .build()

            return rv.SceneConfig.builder() \
                .with_id(id) \
                .with_task(task) \
                .with_raster_source(raster_source) \
                .with_label_source(label_source) \
                .build()

        train_scenes = [make_scene(id) for id in train_scene_ids]
        val_scenes = [make_scene(id) for id in val_scene_ids]

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes(train_scenes) \
                                  .with_validation_scenes(val_scenes) \
                                  .build()

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id(exp_id) \
                                        .with_root_uri(root_uri) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .build()

        return experiment


if __name__ == '__main__':
    rv.main()
