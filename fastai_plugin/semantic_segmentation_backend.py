from os.path import join, basename, dirname, isfile
import uuid
import zipfile
import csv
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from fastai.vision import (
    SegmentationItemList, get_transforms, models, unet_learner, Image)
from fastai.callbacks import SaveModelCallback, CSVLogger, Callback
from fastai.basic_train import load_learner

from rastervision.utils.files import (
        get_local_path, make_dir, upload_or_copy, list_paths,
        download_if_needed, sync_from_dir, sync_to_dir)
from rastervision.utils.misc import save_img
from rastervision.backend import Backend
from rastervision.data.label import SemanticSegmentationLabels
from rastervision.data.label_source.utils import color_to_triple


class SyncCallback(Callback):
    def __init__(self, from_dir, to_uri, sync_interval=1):
        self.from_dir = from_dir
        self.to_uri = to_uri
        self.sync_interval = sync_interval

    def on_epoch_end(self, **kwargs):
        if (kwargs['epoch'] + 1) % self.sync_interval == 0:
            sync_to_dir(self.from_dir, self.to_uri)


class MyCSVLogger(CSVLogger):
    """Custom CSVLogger

    Modified to:
    - flush after each epoch
    - append to log if already exists
    - use start_epoch
    """
    def __init__(self, learn, filename='history', start_epoch=0):
        super().__init__(learn, filename)
        self.start_epoch = start_epoch

    def on_train_begin(self, **kwargs):
        if self.path.exists():
            self.file = self.path.open('a')
        else:
            super().on_train_begin(**kwargs)

    def on_epoch_end(self, epoch, smooth_loss, last_metrics, **kwargs):
        effective_epoch = self.start_epoch + epoch
        out = super().on_epoch_end(
            effective_epoch, smooth_loss, last_metrics, **kwargs)
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


def semseg_acc(input, target):
    # Note: custom metrics need to be at global level for learner to be saved.
    nodata_id = 0
    target = target.squeeze(1)
    mask = target != nodata_id
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()


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
    def __init__(self, config, task_config):
        # chip_uri, model_uri, train_uri
        self.config = config
        self.task_config = task_config
        self.inf_learner = None

        # TODO get logging to work for plugins
        print('Backend config')
        print('--------------')
        for k, v in self.config.items():
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
        chip_uri = self.config['chip_uri']
        group = str(uuid.uuid4())
        group_uri = join(chip_uri, '{}.zip'.format(group))
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
        # Setup hyperparams.
        bs = int(self.config.get('bs', 8))
        wd = self.config.get('wd', 1e-2)
        lr = self.config.get('lr', 2e-3)
        num_epochs = int(self.config.get('num_epochs', 10))
        model_arch = self.config.get('model_arch', 'resnet50')
        model_arch = getattr(models, model_arch)
        fp16 = self.config.get('fp16', False)
        sync_interval = self.config.get('sync_interval', 1)
        debug = self.config.get('debug', False)

        chip_uri = self.config['chip_uri']
        train_uri = self.config['train_uri']

        # Sync output of previous training run from cloud.
        train_dir = get_local_path(train_uri, tmp_dir)
        make_dir(train_dir)
        sync_from_dir(train_uri, train_dir)

        # Get zip file for each group, and unzip them into chip_dir.
        chip_dir = join(tmp_dir, 'chips')
        make_dir(chip_dir)
        for zip_uri in list_paths(chip_uri, 'zip'):
            zip_path = download_if_needed(zip_uri, tmp_dir)
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(chip_dir)

        # Setup data loader.
        def get_label_path(im_path):
            return Path(str(im_path.parent)[:-4] + '-labels') / im_path.name

        size = self.task_config.chip_size
        class_map = self.task_config.class_map
        classes = ['nodata'] + class_map.get_class_names()
        data = (SegmentationItemList.from_folder(chip_dir)
                .split_by_folder(train='train-img', valid='val-img')
                .label_from_func(get_label_path, classes=classes)
                .transform(get_transforms(), size=size, tfm_y=True)
                .databunch(bs=bs))
        print(data)

        if debug:
            # We make debug chips during the run-time of the train command
            # rather than the chip command
            # because this is a better test (see "visualize just before the net"
            # in https://karpathy.github.io/2019/04/25/recipe/), and because
            # it's more convenient since we have the databunch here.
            make_debug_chips(data, class_map, train_dir)

        # Setup learner.
        metrics = [semseg_acc]
        learn = unet_learner(data, model_arch, metrics=metrics, wd=wd, bottle=True)
        learn.unfreeze()

        if fp16 and torch.cuda.is_available():
            # This loss_scale works for Resnet 34 and 50. You might need to adjust this
            # for other models.
            learn = learn.to_fp16(loss_scale=256)

        # Setup ability to resume training if model exists.
        # This hack won't properly set the learning as a function of epochs
        # when resuming.
        learner_path = join(train_dir, 'learner.pth')
        log_path = join(train_dir, 'log')

        start_epoch = 0
        if isfile(learner_path):
            print('Loading saved model...')
            start_epoch = get_last_epoch(str(log_path) + '.csv') + 1
            if start_epoch >= num_epochs:
                print('Training is already done. If you would like to re-train'
                      ', delete the previous results of training in '
                      '{}.'.format(train_uri))
                return

            learn.load(learner_path[:-4])
            print('Resuming from epoch {}'.format(start_epoch))
            print('Note: fastai does not support a start_epoch, so epoch 0 below '
                  'corresponds to {}'.format(start_epoch))
        epochs_left = num_epochs - start_epoch

        # Setup callbacks and train model.
        callbacks = [
            SaveModelCallback(learn, name=learner_path[:-4]),
            MyCSVLogger(learn, filename=log_path, start_epoch=start_epoch),
            SyncCallback(train_dir, train_uri, sync_interval)
        ]
        learn.fit(epochs_left, lr, callbacks=callbacks)

        # Export model for inference
        model_uri = self.config['model_uri']
        model_path = get_local_path(model_uri, tmp_dir)
        learn.export(model_path)

        # Sync output to cloud.
        sync_to_dir(train_dir, train_uri)

    def load_model(self, tmp_dir):
        """Load the model in preparation for one or more prediction calls."""
        if self.inf_learner is None:
            model_uri = self.config['model_uri']
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