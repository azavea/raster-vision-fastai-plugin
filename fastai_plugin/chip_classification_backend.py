import os
from os.path import join, basename, dirname, isfile
import uuid
import zipfile
import glob
from pathlib import Path
import random
import shutil
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from fastai.vision import (ImageList, get_transforms, models, cnn_learner,
                           Image, ImageSegment)
from fastai.callbacks import CSVLogger, TrackEpochCallback
from fastai.basic_train import load_learner
from fastai.basic_data import DatasetType
from fastai.vision.transform import dihedral
from torch.utils.data.sampler import WeightedRandomSampler

from rastervision.utils.files import (get_local_path, make_dir, upload_or_copy,
                                      list_paths, download_if_needed,
                                      sync_from_dir, sync_to_dir, str_to_file)
from rastervision.utils.misc import save_img
from rastervision.backend import Backend
from rastervision.data.label import ChipClassificationLabels
from rastervision.data.label_source.utils import color_to_triple

from fastai_plugin.utils import (SyncCallback, MySaveModelCallback,
                                 ExportCallback, MyCSVLogger, Precision,
                                 Recall, FBeta, zipdir)

log = logging.getLogger(__name__)


def make_debug_chips(data,
                     class_map,
                     tmp_dir,
                     train_uri,
                     debug_prob=.5,
                     count=20):
    def _make_debug_chips(split):
        debug_chips_dir = join(tmp_dir, '{}-debug-chips'.format(split))
        zip_path = join(tmp_dir, '{}-debug-chips.zip'.format(split))
        zip_uri = join(train_uri, '{}-debug-chips.zip'.format(split))
        make_dir(debug_chips_dir)
        ds = data.train_ds if split == 'train' else data.valid_ds
        n = 0
        for i, (x, y) in enumerate(ds):
            if n >= count:
                break
            if random.uniform(0, 1) < debug_prob:
                x.show(y=y)
                plt.savefig(
                    join(debug_chips_dir, '{}.png'.format(i)), figsize=(5, 5))
                plt.close()
            n += 1
        zipdir(debug_chips_dir, zip_path)
        upload_or_copy(zip_path, zip_uri)

    _make_debug_chips('train')
    _make_debug_chips('val')


def merge_class_dirs(scene_class_dirs, output_dir):
    seen_classes = set([])
    chip_ind = 0
    for scene_class_dir in scene_class_dirs:
        for class_name, src_class_dir in scene_class_dir.items():
            dst_class_dir = join(output_dir, class_name)
            if class_name not in seen_classes:
                make_dir(dst_class_dir)
                seen_classes.add(class_name)

            for src_class_file in [
                    join(src_class_dir, class_file)
                    for class_file in os.listdir(src_class_dir)
            ]:
                dst_class_file = join(dst_class_dir, '{}.png'.format(chip_ind))
                shutil.move(src_class_file, dst_class_file)
                chip_ind += 1


class FileGroup(object):
    def __init__(self, base_uri, tmp_dir):
        self.tmp_dir = tmp_dir
        self.base_uri = base_uri
        self.base_dir = self.get_local_path(base_uri)

        make_dir(self.base_dir)

    def get_local_path(self, uri):
        return get_local_path(uri, self.tmp_dir)

    def upload_or_copy(self, uri):
        upload_or_copy(self.get_local_path(uri), uri)

    def download_if_needed(self, uri):
        return download_if_needed(uri, self.tmp_dir)


class DatasetFiles(FileGroup):
    """Utilities for files produced when calling convert_training_data."""

    def __init__(self, base_uri, tmp_dir):
        FileGroup.__init__(self, base_uri, tmp_dir)

        self.partition_id = uuid.uuid4()

        self.training_zip_uri = join(
            base_uri, 'training-{}.zip'.format(self.partition_id))
        self.training_local_uri = join(self.base_dir,
                                       'training-{}'.format(self.partition_id))
        self.training_download_uri = self.get_local_path(
            join(self.base_uri, 'training'))
        make_dir(self.training_local_uri)

        self.validation_zip_uri = join(
            base_uri, 'validation-{}.zip'.format(self.partition_id))
        self.validation_local_uri = join(
            self.base_dir, 'validation-{}'.format(self.partition_id))
        self.validation_download_uri = self.get_local_path(
            join(self.base_uri, 'validation'))
        make_dir(self.validation_local_uri)

    def download(self):
        def _download(split, output_dir):
            scene_class_dirs = []
            for uri in list_paths(self.base_uri, 'zip'):
                base_name = os.path.basename(uri)
                if base_name.startswith(split):
                    data_zip_path = self.download_if_needed(uri)
                    data_dir = os.path.splitext(data_zip_path)[0]
                    shutil.unpack_archive(data_zip_path, data_dir)

                    # Append each of the directories containing this partitions'
                    # labeled images based on the class directory.
                    data_dir_subdirectories = next(os.walk(data_dir))[1]
                    scene_class_dirs.append(
                        dict([(class_name, os.path.join(data_dir, class_name))
                              for class_name in data_dir_subdirectories]))
            merge_class_dirs(scene_class_dirs, output_dir)

        _download('training', self.training_download_uri)
        _download('validation', self.validation_download_uri)

    def upload(self):
        def _upload(data_dir, zip_uri, split):
            if not any(os.scandir(data_dir)):
                log.warn(
                    'No data to write for split {} in partition {}...'.format(
                        split, self.partition_id))
            else:
                shutil.make_archive(data_dir, 'zip', data_dir)
                upload_or_copy(data_dir + '.zip', zip_uri)

        _upload(self.training_local_uri, self.training_zip_uri, 'training')
        _upload(self.validation_local_uri, self.validation_zip_uri,
                'validation')


class ChipClassificationBackend(Backend):
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

        This writes 
        {tmp_dir}/scratch-{uuid}/
            {scene_id}-{uuid}/
                {class_name}/
                    {chip_idx}.png

        Args:
            scene: Scene
            data: TrainingData

        Returns:
            backend-specific data-structures consumed by backend's
            process_sceneset_results
        """

        scratch_dir = join(tmp_dir, 'scratch-{}'.format(uuid.uuid4()))
        # Ensure directory is unique since scene id's could be shared between
        # training and test sets.
        scene_dir = join(scratch_dir, '{}-{}'.format(scene.id, uuid.uuid4()))
        class_dirs = {}

        for chip_idx, (chip, window, labels) in enumerate(data):
            class_id = labels.get_cell_class_id(window)
            # If a chip is not associated with a class, don't
            # use it in training data.
            if class_id is None:
                continue
            class_name = self.task_config.class_map.get_by_id(class_id).name
            class_dir = join(scene_dir, class_name)
            make_dir(class_dir)
            class_dirs[class_name] = class_dir
            chip_name = '{}.png'.format(chip_idx)
            chip_path = join(class_dir, chip_name)
            save_img(chip, chip_path)

        return class_dirs

    def process_sceneset_results(self, training_results, validation_results,
                                 tmp_dir):
        """After all scenes have been processed, process the result set.

        This writes two zip files:
            training-{uuid}.zip
            validation-{uuid}.zip
        each containing
            {class_name}/{chip_idx}.png

        Args:
            training_results: dependent on the ml_backend's process_scene_data
            validation_results: dependent on the ml_backend's
                process_scene_data
        """
        self.print_options()

        dataset_files = DatasetFiles(self.backend_opts.chip_uri, tmp_dir)
        training_dir = dataset_files.training_local_uri
        validation_dir = dataset_files.validation_local_uri

        merge_class_dirs(training_results, training_dir)
        merge_class_dirs(validation_results, validation_dir)
        dataset_files.upload()

    def train(self, tmp_dir):
        """Train a model."""
        self.print_options()

        # Sync output of previous training run from cloud.
        train_uri = self.backend_opts.train_uri
        train_dir = get_local_path(train_uri, tmp_dir)
        make_dir(train_dir)
        sync_from_dir(train_uri, train_dir)
        '''
            Get zip file for each group, and unzip them into chip_dir in a
            way that works well with FastAI.

            The resulting directory structure would be:
            <chip_dir>/
                train/
                    training-<uuid1>/
                        <class1>/
                            ...
                        <class2>/
                            ...
                        ...
                    training-<uuid2>/
                        <class1>/
                            ...
                        <class2>/
                            ...
                        ...
                    ...
                val/
                    validation-<uuid1>/
                        <class1>/
                            ...
                        <class2>/
                            ...
                        ...
                    validation-<uuid2>/
                        <class1>/
                            ...
                        <class2>/
                            ...
                        ...
                    ...

        '''
        chip_dir = join(tmp_dir, 'chips/')
        make_dir(chip_dir)
        for zip_uri in list_paths(self.backend_opts.chip_uri, 'zip'):
            zip_name = Path(zip_uri).name
            if zip_name.startswith('train'):
                extract_dir = chip_dir + 'train/'
            elif zip_name.startswith('val'):
                extract_dir = chip_dir + 'val/'
            else:
                continue
            zip_path = download_if_needed(zip_uri, tmp_dir)
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(extract_dir)

        # Setup data loader.
        def get_label_path(im_path):
            return Path(str(im_path.parent)[:-4] + '-labels') / im_path.name

        size = self.task_config.chip_size
        class_map = self.task_config.class_map
        classes = class_map.get_class_names()
        num_workers = 0 if self.train_opts.debug else 4
        tfms = get_transforms(flip_vert=self.train_opts.flip_vert)

        def get_data(train_sampler=None):
            data = (ImageList.from_folder(chip_dir).split_by_folder(
                train='train', valid='val').label_from_folder().transform(
                    tfms, size=size).databunch(
                        bs=self.train_opts.batch_sz,
                        num_workers=num_workers,
                    ))
            return data

        data = get_data()

        if self.train_opts.debug:
            make_debug_chips(data, class_map, tmp_dir, train_uri)

        # Setup learner.
        ignore_idx = -1
        metrics = [
            Precision(average='weighted', clas_idx=1, ignore_idx=ignore_idx),
            Recall(average='weighted', clas_idx=1, ignore_idx=ignore_idx),
            FBeta(
                average='weighted', clas_idx=1, beta=1, ignore_idx=ignore_idx)
        ]
        model_arch = getattr(models, self.train_opts.model_arch)
        learn = cnn_learner(
            data,
            model_arch,
            metrics=metrics,
            wd=self.train_opts.weight_decay,
            path=train_dir)

        learn.unfreeze()

        if self.train_opts.fp16 and torch.cuda.is_available():
            # This loss_scale works for Resnet 34 and 50. You might need to adjust this
            # for other models.
            learn = learn.to_fp16(loss_scale=256)

        # Setup callbacks and train model.
        model_path = get_local_path(self.backend_opts.model_uri, tmp_dir)

        pretrained_uri = self.backend_opts.pretrained_uri
        if pretrained_uri:
            print('Loading weights from pretrained_uri: {}'.format(
                pretrained_uri))
            pretrained_path = download_if_needed(pretrained_uri, tmp_dir)
            learn.model.load_state_dict(
                torch.load(pretrained_path, map_location=learn.data.device),
                strict=False)

        # Save every epoch so that resume functionality provided by
        # TrackEpochCallback will work.
        callbacks = [
            TrackEpochCallback(learn),
            MySaveModelCallback(learn, every='epoch'),
            MyCSVLogger(learn, filename='log'),
            ExportCallback(learn, model_path, monitor='f_beta'),
            SyncCallback(train_dir, self.backend_opts.train_uri,
                         self.train_opts.sync_interval)
        ]

        lr = self.train_opts.lr
        num_epochs = self.train_opts.num_epochs
        if self.train_opts.one_cycle:
            if lr is None:
                learn.lr_find()
                learn.recorder.plot(suggestion=True, return_fig=True)
                lr = learn.recorder.min_grad_lr
                print('lr_find() found lr: {}'.format(lr))
            learn.fit_one_cycle(num_epochs, lr, callbacks=callbacks)
        else:
            learn.fit(num_epochs, lr, callbacks=callbacks)

        # Since model is exported every epoch, we need some other way to
        # show that training is finished.
        str_to_file('done!', self.backend_opts.train_done_uri)

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
            self.device = torch.device("cuda:0" if torch.cuda.
                                       is_available() else "cpu")

    def predict(self, chips, windows, tmp_dir):
        """Return predictions for a chip using model.

        Args:
            chips: [[height, width, channels], ...] numpy array of chips
            windows: List of boxes that are the windows aligned with the chips.

        Return:
            Labels object containing predictions
        """
        self.load_model(tmp_dir)

        # (batch_size, h, w, nchannels) --> (batch_size, nchannels, h, w)
        chips = torch.Tensor(chips).permute((0, 3, 1, 2)) / 255.
        chips = chips.to(self.device)

        model = self.inf_learner.model.eval()
        preds = model(chips).detach().cpu()

        labels = ChipClassificationLabels()

        for class_probs, window in zip(preds, windows):
            # Add 1 to class_id since they start at 1.
            class_id = int(class_probs.argmax() + 1)

            labels.set_cell(window, class_id, class_probs)

        return labels
