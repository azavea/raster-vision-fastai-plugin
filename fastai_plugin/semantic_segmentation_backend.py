from os.path import join, basename, dirname, isfile
import uuid
import zipfile
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from fastai.vision import (
    SegmentationItemList, get_transforms, models, unet_learner, Image)
from fastai.callbacks import SaveModelCallback, CSVLogger, TrackEpochCallback
from fastai.basic_train import load_learner

from rastervision.utils.files import (
        get_local_path, make_dir, upload_or_copy, list_paths,
        download_if_needed, sync_from_dir, sync_to_dir)
from rastervision.utils.misc import save_img
from rastervision.backend import Backend
from rastervision.data.label import SemanticSegmentationLabels
from rastervision.data.label_source.utils import color_to_triple

from fastai_plugin.utils import (
    SyncCallback, ExportCallback, MyCSVLogger, get_last_epoch,
    Precision, Recall, FBeta)


def make_debug_chips(data, class_map, train_dir):
    # TODO get rid of white frame
    # TODO zip them
    # use grey for nodata
    colors = [class_map.get_by_id(i).color
              for i in range(1, len(class_map) + 1)]
    colors = ['grey'] + colors
    colors = [color_to_triple(c) for c in colors]
    colors = [tuple([x / 255 for x in c]) for c in colors]
    cmap = matplotlib.colors.ListedColormap(colors)

    def _make_debug_chips(split):
        debug_chips_dir = join(train_dir, '{}-debug-chips'.format(split))
        make_dir(debug_chips_dir)
        ds = data.train_ds if split == 'train' else data.valid_ds
        for i, (x, y) in enumerate(ds):
            plt.axis('off')
            plt.imshow(x.data.permute((1, 2, 0)).numpy())
            plt.imshow(y.data.squeeze().numpy(), alpha=0.4, vmin=0,
                       vmax=len(colors), cmap=cmap)
            plt.savefig(join(debug_chips_dir, '{}.png'.format(i)),
                        figsize=(3, 3))
            plt.close()
    _make_debug_chips('train')
    _make_debug_chips('val')


class SemanticSegmentationBackend(Backend):
    def __init__(self, task_config, backend_opts, train_opts):
        self.task_config = task_config
        self.backend_opts = backend_opts
        self.train_opts = train_opts
        self.inf_learner = None

    def print_options(self):
        # TODO get logging to work for plugins
        print('Backend options')
        print('--------------')
        for k, v in self.backend_opts.__dict__.items():
            print('{}: {}'.format(k, v))
        print()

        print('Train options')
        print('--------------')
        for k, v in self.train_opts.__dict__.items():
            print('{}: {}'.format(k, v))
        print()

    def process_scene_data(self, scene, data, tmp_dir):
        """Process each scene's training data.

        This writes {scene_id}/img/{scene_id}-{ind}.png and
        {scene_id}/labels/{scene_id}-{ind}.png

        Args:
            scene: Scene
            data: TrainingData

        Returns:
            backend-specific data-structures consumed by backend's
            process_sceneset_results
        """
        scene_dir = join(tmp_dir, str(scene.id))
        img_dir = join(scene_dir, 'img')
        labels_dir = join(scene_dir, 'labels')

        make_dir(img_dir)
        make_dir(labels_dir)

        for ind, (chip, window, labels) in enumerate(data):
            chip_path = join(img_dir, '{}-{}.png'.format(scene.id, ind))
            label_path = join(labels_dir, '{}-{}.png'.format(scene.id, ind))
            save_img(chip, chip_path)
            label_im = labels.get_label_arr(window).astype(np.uint8)
            save_img(label_im, label_path)

        return scene_dir

    def process_sceneset_results(self, training_results, validation_results,
                                 tmp_dir):
        """After all scenes have been processed, process the result set.

        This writes a zip file for a group of scenes at {chip_uri}/{uuid}.zip
        containing:
        train-img/{scene_id}-{ind}.png
        train-labels/{scene_id}-{ind}.png
        val-img/{scene_id}-{ind}.png
        val-labels/{scene_id}-{ind}.png

        Args:
            training_results: dependent on the ml_backend's process_scene_data
            validation_results: dependent on the ml_backend's
                process_scene_data
        """
        self.print_options()

        group = str(uuid.uuid4())
        group_uri = join(self.backend_opts.chip_uri, '{}.zip'.format(group))
        group_path = get_local_path(group_uri, tmp_dir)
        make_dir(group_path, use_dirname=True)

        with zipfile.ZipFile(group_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            def _write_zip(results, split):
                for scene_dir in results:
                    scene_paths = glob.glob(join(scene_dir, '**/*.png'))
                    for p in scene_paths:
                        zipf.write(p, join(
                            '{}-{}'.format(
                                split,
                                dirname(p).split('/')[-1]),
                            basename(p)))
            _write_zip(training_results, 'train')
            _write_zip(validation_results, 'val')

        upload_or_copy(group_path, group_uri)

    def train(self, tmp_dir):
        """Train a model."""
        self.print_options()

        # Sync output of previous training run from cloud.
        train_dir = get_local_path(self.backend_opts.train_uri, tmp_dir)
        make_dir(train_dir)
        sync_from_dir(self.backend_opts.train_uri, train_dir)

        # Get zip file for each group, and unzip them into chip_dir.
        chip_dir = join(tmp_dir, 'chips')
        make_dir(chip_dir)
        for zip_uri in list_paths(self.backend_opts.chip_uri, 'zip'):
            zip_path = download_if_needed(zip_uri, tmp_dir)
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(chip_dir)

        # Setup data loader.
        def get_label_path(im_path):
            return Path(str(im_path.parent)[:-4] + '-labels') / im_path.name

        size = self.task_config.chip_size
        class_map = self.task_config.class_map
        classes = ['nodata'] + class_map.get_class_names()
        num_workers = 0 if self.train_opts.debug else 4
        data = (SegmentationItemList.from_folder(chip_dir)
                .split_by_folder(train='train-img', valid='val-img')
                .label_from_func(get_label_path, classes=classes)
                .transform(get_transforms(), size=size, tfm_y=True)
                .databunch(bs=self.train_opts.batch_sz,
                           num_workers=num_workers))
        print(data)

        if self.train_opts.debug:
            # We make debug chips during the run-time of the train command
            # rather than the chip command
            # because this is a better test (see "visualize just before the net"
            # in https://karpathy.github.io/2019/04/25/recipe/), and because
            # it's more convenient since we have the databunch here.
            make_debug_chips(data, class_map, train_dir)

        # Setup learner.
        ignore_idx = 0
        metrics = [
            Precision(average='weighted', clas_idx=1, ignore_idx=ignore_idx),
            Recall(average='weighted', clas_idx=1, ignore_idx=ignore_idx),
            FBeta(average='weighted', clas_idx=1, beta=1, ignore_idx=ignore_idx)]
        model_arch = getattr(models, self.train_opts.model_arch)
        learn = unet_learner(
            data, model_arch, metrics=metrics, wd=self.train_opts.weight_decay,
            bottle=True, path=train_dir)
        learn.unfreeze()

        if self.train_opts.fp16 and torch.cuda.is_available():
            # This loss_scale works for Resnet 34 and 50. You might need to adjust this
            # for other models.
            learn = learn.to_fp16(loss_scale=256)

        # Setup callbacks and train model.
        model_path = get_local_path(self.backend_opts.model_uri, tmp_dir)
        print(model_path)
        callbacks = [
            TrackEpochCallback(learn),
            SaveModelCallback(learn, every='epoch'),
            MyCSVLogger(learn, filename='log'),
            ExportCallback(learn, model_path),
            SyncCallback(train_dir, self.backend_opts.train_uri,
                         self.train_opts.sync_interval)
        ]
        learn.fit(self.train_opts.num_epochs, self.train_opts.lr,
                  callbacks=callbacks)

        # Sync output to cloud.
        sync_to_dir(train_dir, self.backend_opts.train_uri)

    def load_model(self, tmp_dir):
        """Load the model in preparation for one or more prediction calls."""
        if self.inf_learner is None:
            self.print_options()
            model_uri = self.backend_opts.model_uri
            model_path = download_if_needed(model_uri, tmp_dir)
            self.inf_learner = load_learner(
                dirname(model_path), basename(model_path))

    def predict(self, chips, windows, tmp_dir):
        """Return predictions for a chip using model.

        Args:
            chips: [[height, width, channels], ...] numpy array of chips
            windows: List of boxes that are the windows aligned with the chips.

        Return:
            Labels object containing predictions
        """
        self.load_model(tmp_dir)
        # TODO get it to work on a whole batch at a time
        chip = torch.Tensor(chips[0]).permute((2, 0, 1)) / 255.
        im = Image(chip)

        label_arr = self.inf_learner.predict(im)[1].squeeze().numpy()

        # Return "trivial" instance of SemanticSegmentationLabels that holds a single
        # window and has ability to get labels for that one window.
        def label_fn(_window):
            if _window == windows[0]:
                return label_arr
            else:
                raise ValueError('Trying to get labels for unknown window.')

        return SemanticSegmentationLabels(windows, label_fn)