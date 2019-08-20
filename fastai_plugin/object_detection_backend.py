from os.path import join, basename, dirname, isfile
import uuid
import zipfile
import glob
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from fastai.vision import (
    ObjectItemList, bb_pad_collate, get_transforms, models,
    Image, get_annotations)
from fastai.callbacks import SaveModelCallback, CSVLogger, TrackEpochCallback
from fastai.basic_train import load_learner, Learner

from rastervision.utils.files import (
    get_local_path, make_dir, upload_or_copy, list_paths,
    download_if_needed, sync_from_dir, sync_to_dir, str_to_file,
    json_to_file)
from rastervision.utils.misc import save_img
from rastervision.backend import Backend
from rastervision.data import ObjectDetectionLabels

from fastai_plugin.utils import (
    SyncCallback, ExportCallback, MyCSVLogger,
    Precision, Recall, FBeta, zipdir)
from fastai_plugin.retinanet import (
    create_body, RetinaNet, RetinaNetFocalLoss, retina_net_split,
    get_predictions, show_results)


def make_debug_chips(data, class_map, tmp_dir, train_uri, debug_prob=1.0):
    def _make_debug_chips(split):
        debug_chips_dir = join(tmp_dir, '{}-debug-chips'.format(split))
        zip_path = join(tmp_dir, '{}-debug-chips.zip'.format(split))
        zip_uri = join(train_uri, '{}-debug-chips.zip'.format(split))
        make_dir(debug_chips_dir)
        ds = data.train_ds if split == 'train' else data.valid_ds
        for i, (x, y) in enumerate(ds):
            if random.uniform(0, 1) < debug_prob:
                x.show(y=y)
                plt.savefig(join(debug_chips_dir, '{}.png'.format(i)),
                            figsize=(3, 3))
                plt.close()
        zipdir(debug_chips_dir, zip_path)
        upload_or_copy(zip_path, zip_uri)

    _make_debug_chips('train')
    _make_debug_chips('val')


class ObjectDetectionBackend(Backend):
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

        This writes {scene_id}/{scene_id}-{ind}.png and
        {scene_id}/{scene_id}-labels.json in COCO format.

        Args:
            scene: Scene
            data: TrainingData

        Returns:
            backend-specific data-structures consumed by backend's
            process_sceneset_results
        """

        scene_dir = join(tmp_dir, str(scene.id))
        labels_path = join(scene_dir, '{}-labels.json'.format(scene.id))

        make_dir(scene_dir)
        images = []
        annotations = []
        categories = [{'id': item.id, 'name': item.name}
                      for item in self.task_config.class_map.get_items()]

        for im_ind, (chip, window, labels) in enumerate(data):
            im_id = '{}-{}'.format(scene.id, im_ind)
            fn = '{}.png'.format(im_id)
            chip_path = join(scene_dir, fn)
            save_img(chip, chip_path)
            images.append({
                'file_name': fn,
                'id': im_id,
                'height': chip.shape[0],
                'width': chip.shape[1]
            })

            npboxes = labels.get_npboxes()
            npboxes = ObjectDetectionLabels.global_to_local(npboxes, window)
            for box_ind, (box, class_id) in enumerate(
                    zip(npboxes, labels.get_class_ids())):
                bbox = [box[1], box[0], box[3]-box[1], box[2]-box[0]]
                bbox = [int(i) for i in bbox]
                annotations.append({
                    'id': '{}-{}'.format(im_id, box_ind),
                    'image_id': im_id,
                    'bbox': bbox,
                    'category_id': int(class_id)
                })

        coco_dict = {
            'images': images,
            'annotations': annotations,
            'categories': categories
        }
        json_to_file(coco_dict, labels_path)

        return scene_dir

    def process_sceneset_results(self, training_results, validation_results,
                                 tmp_dir):
        """After all scenes have been processed, process the result set.

        This writes a zip file for a group of scenes at {chip_uri}/{uuid}.zip
        containing:
        train/{scene_id}-{ind}.png
        train/{scene_id}-labels.json
        val/{scene_id}-{ind}.png
        val/{scene_id}-labels.json

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
                    scene_paths = glob.glob(join(scene_dir, '*'))
                    for p in scene_paths:
                        zipf.write(p, join(split, basename(p)))
            _write_zip(training_results, 'train')
            _write_zip(validation_results, 'valid')

        upload_or_copy(group_path, group_uri)

    def train(self, tmp_dir):
        """Train a model."""
        self.print_options()

        # Sync output of previous training run from cloud.
        train_uri = self.backend_opts.train_uri
        train_dir = get_local_path(train_uri, tmp_dir)
        make_dir(train_dir)
        sync_from_dir(train_uri, train_dir)

        # Get zip file for each group, and unzip them into chip_dir.
        chip_dir = join(tmp_dir, 'chips')
        make_dir(chip_dir)
        for zip_uri in list_paths(self.backend_opts.chip_uri, 'zip'):
            zip_path = download_if_needed(zip_uri, tmp_dir)
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(chip_dir)

        # Setup data loader.
        train_images = []
        train_lbl_bbox = []
        for annotation_path in glob.glob(join(chip_dir, 'train/*.json')):
            images, lbl_bbox = get_annotations(annotation_path)
            train_images += images
            train_lbl_bbox += lbl_bbox

        val_images = []
        val_lbl_bbox = []
        for annotation_path in glob.glob(join(chip_dir, 'valid/*.json')):
            images, lbl_bbox = get_annotations(annotation_path)
            val_images += images
            val_lbl_bbox += lbl_bbox

        images = train_images + val_images
        lbl_bbox = train_lbl_bbox + val_lbl_bbox

        img2bbox = dict(zip(images, lbl_bbox))
        get_y_func = lambda o: img2bbox[o.name]
        num_workers = 0 if self.train_opts.debug else 4
        data = ObjectItemList.from_folder(chip_dir)
        data = data.split_by_folder()
        data = data.label_from_func(get_y_func)
        data = data.transform(
            get_transforms(), size=self.task_config.chip_size, tfm_y=True)
        data = data.databunch(
            bs=self.train_opts.batch_sz, collate_fn=bb_pad_collate,
            num_workers=num_workers)
        print(data)

        if self.train_opts.debug:
            make_debug_chips(
                data, self.task_config.class_map, tmp_dir, train_uri)

        # Setup callbacks and train model.
        ratios = [1/2, 1, 2]
        scales = [1, 2**(-1/3), 2**(-2/3)]
        model_arch = getattr(models, self.train_opts.model_arch)
        encoder = create_body(model_arch, cut=-2)
        model = RetinaNet(encoder, data.c, final_bias=-4)
        crit = RetinaNetFocalLoss(scales=scales, ratios=ratios)
        learn = Learner(data, model, loss_func=crit, path=train_dir)
        learn = learn.split(retina_net_split)

        model_path = get_local_path(self.backend_opts.model_uri, tmp_dir)

        pretrained_uri = self.backend_opts.pretrained_uri
        if pretrained_uri:
            print('Loading weights from pretrained_uri: {}'.format(
                pretrained_uri))
            pretrained_path = download_if_needed(pretrained_uri, tmp_dir)
            learn.load(pretrained_path[:-4])

        callbacks = [
            TrackEpochCallback(learn),
            SaveModelCallback(learn, every='epoch'),
            MyCSVLogger(learn, filename='log'),
            ExportCallback(learn, model_path),
            SyncCallback(train_dir, self.backend_opts.train_uri,
                         self.train_opts.sync_interval)
        ]
        learn.unfreeze()
        learn.fit(self.train_opts.num_epochs, self.train_opts.lr,
                  callbacks=callbacks)

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
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def predict(self, chips, windows, tmp_dir):
        """Return predictions for a chip using model.

        Args:
            chips: [[height, width, channels], ...] numpy array of chips
            windows: List of boxes that are the windows aligned with the chips.

        Return:
            Labels object containing predictions
        """
        self.load_model(tmp_dir)

        '''
        # Sync output of previous training run from cloud.
        train_uri = self.backend_opts.train_uri
        train_dir = get_local_path(train_uri, tmp_dir)
        make_dir(train_dir)
        sync_from_dir(train_uri, train_dir)

        # Get zip file for each group, and unzip them into chip_dir.
        chip_dir = join(tmp_dir, 'chips')
        make_dir(chip_dir)
        for zip_uri in list_paths(self.backend_opts.chip_uri, 'zip'):
            zip_path = download_if_needed(zip_uri, tmp_dir)
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(chip_dir)

        # Setup data loader.
        train_images = []
        train_lbl_bbox = []
        for annotation_path in glob.glob(join(chip_dir, 'train/*.json')):
            images, lbl_bbox = get_annotations(annotation_path)
            train_images += images
            train_lbl_bbox += lbl_bbox

        val_images = []
        val_lbl_bbox = []
        for annotation_path in glob.glob(join(chip_dir, 'valid/*.json')):
            images, lbl_bbox = get_annotations(annotation_path)
            val_images += images
            val_lbl_bbox += lbl_bbox

        images = train_images + val_images
        lbl_bbox = train_lbl_bbox + val_lbl_bbox

        img2bbox = dict(zip(images, lbl_bbox))
        get_y_func = lambda o: img2bbox[o.name]
        num_workers = 0 if self.train_opts.debug else 4
        data = ObjectItemList.from_folder(chip_dir)
        data = data.split_by_folder()
        data = data.label_from_func(get_y_func)
        data = data.transform(
            get_transforms(), size=self.task_config.chip_size, tfm_y=True)
        data = data.databunch(
            bs=self.train_opts.batch_sz, collate_fn=bb_pad_collate,
            num_workers=num_workers)
        print(data)

        self.inf_learner.data = data
        show_results(self.inf_learner, start=0, n=4, detect_thresh=0.5,
                     figsize=(10, 25))
        plt.savefig('/opt/data/tmp')
        '''

        #####

        labels = ObjectDetectionLabels.make_empty()
        chips = torch.Tensor(chips).permute((0, 3, 1, 2)) / 255.
        chips = chips.to(self.device)
        model = self.inf_learner.model.eval()

        output = model(chips)

        for chip_ind, (chip, window) in enumerate(zip(chips, windows)):
            boxes, class_ids, scores = get_predictions(
                output, chip_ind, detect_thresh=0.2)
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu()
                class_ids = class_ids.cpu()
                scores = scores.cpu()
                t_sz = torch.Tensor([window.get_height(), window.get_width()])[None].float()
                boxes[:,:2] = boxes[:,:2] - boxes[:,2:]/2
                boxes[:,:2] = (boxes[:,:2] + 1) * t_sz/2
                boxes[:,2:] = boxes[:,2:] * t_sz
                boxes[:, 2:4] = boxes[:, 0:2] + boxes[:, 2:4] / 2

                boxes = boxes.detach().numpy().astype(np.float)
                class_ids = class_ids.detach().numpy()
                scores = scores.detach().numpy()

                boxes = ObjectDetectionLabels.local_to_global(
                    boxes, window)
                class_ids = class_ids.astype(np.int32) + 1
                labels = (labels + ObjectDetectionLabels(
                    boxes, class_ids, scores=scores))

        return labels
