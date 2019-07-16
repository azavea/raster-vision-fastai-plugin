from os.path import join, basename
import uuid
import zipfile
import glob
from pathlib import Path
import random
import collections
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, Tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import fastai
from fastai.vision import (
    ObjectItemList, bb_pad_collate, get_transforms)
from fastai.callbacks import TrackEpochCallback
from fastai.callback import CallbackHandler
from fastai.basic_train import Learner
from fastai.core import ifnone
from fastai.torch_core import OptLossFunc, OptOptimizer, Optional, Tuple, Union

from rastervision.utils.files import (
    get_local_path, make_dir, upload_or_copy, list_paths,
    download_if_needed, sync_from_dir, sync_to_dir, str_to_file,
    json_to_file)
from rastervision.utils.misc import save_img
from rastervision.backend import Backend
from rastervision.data import ObjectDetectionLabels

from fastai_plugin.utils import (
    SyncCallback, MySaveModelCallback, ExportCallback, MyCSVLogger,
    zipdir)


# Updated this function from fastai so that it injects a bogus bounding box
# with a special label to get around the inability of fastai and torchvision
# to handle images with no ground truth boxes.
def get_annotations(fname, prefix=None):
    "Open a COCO style json in `fname` and returns the lists of filenames (with maybe `prefix`) and labelled bboxes."
    annot_dict = json.load(open(fname))
    id2images, id2bboxes, id2cats = {}, collections.defaultdict(list), collections.defaultdict(list)
    classes = {}
    for o in annot_dict['categories']:
        classes[o['id']] = o['name']

    for o in annot_dict['images']:
        id2images[o['id']] = ifnone(prefix, '') + o['file_name']

    for o in annot_dict['annotations']:
        bb = o['bbox']
        id2bboxes[o['image_id']].append(
            [bb[1], bb[0], bb[3]+bb[1], bb[2]+bb[0]])
        id2cats[o['image_id']].append(classes[o['category_id']])

    # Add a bogus box for each image to ensure that every image has at least
    # one box.
    for o in annot_dict['images']:
        id2bboxes[o['id']].append([0, 0, o['height'], o['width']])
        id2cats[o['id']].append('bogus')

    ids = list(id2images.keys())
    return [id2images[k] for k in ids], [[id2bboxes[k], id2cats[k]] for k in ids]


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


def loss_batch(model:nn.Module, xb:Tensor, yb:Tensor, loss_func:OptLossFunc=None,
               opt:OptOptimizer=None,
               cb_handler:Optional[CallbackHandler]=None)->Tuple[Union[Tensor,int,float,str]]:
    "Calculate loss and metrics for a batch, call out to callbacks as necessary."
    cb_handler = ifnone(cb_handler, CallbackHandler())

    # Translate from fastai box format to torchvision.
    batch_sz = len(xb)
    images = xb
    targets = []
    for i in range(batch_sz):
        boxes = yb[0][i]
        labels = yb[1][i]
        # convert from (ymin, xmin, ymax, xmax) to (xmin, ymin, xmax, ymax)
        boxes = torch.cat((
            boxes[:, 1:2], boxes[:, 0:1], boxes[:, 3:4], boxes[:, 2:3]), dim=1)

        targets.append({
            'boxes': boxes,
            'labels': labels
        })

    # torchvision can only compute losses in training mode,
    # and output in eval mode. So we have to set some things to null
    # values. This isn't ideal and will break some callbacks but I'm not
    # sure how else to handle it at the moment.
    out = None
    loss = torch.Tensor([0.0])
    if model.training:
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
    else:
        out = model(images)

    out = cb_handler.on_loss_begin(out)

    if opt is not None:
        loss,skip_bwd = cb_handler.on_backward_begin(loss)
        if not skip_bwd:                     loss.backward()
        if not cb_handler.on_backward_end(): opt.step()
        if not cb_handler.on_step_end():     opt.zero_grad()

    return loss.detach().cpu()


class ObjectDetectionBackend(Backend):
    def __init__(self, task_config, backend_opts, train_opts):
        self.task_config = task_config
        self.backend_opts = backend_opts
        self.train_opts = train_opts
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        # Add bogus class for handling images with no ground truth boxes.
        # See get_annotations function above.
        categories.append({'id': len(categories)+1, 'name': 'bogus'})
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
        num_classes = data.c
        model = fasterrcnn_resnet50_fpn(
            pretrained=False, progress=True, num_classes=num_classes,
            pretrained_backbone=True, min_size=300, max_size=300)
        learn = Learner(data, model, path=train_dir)
        learn.unfreeze()

        fastai.basic_train.loss_batch = loss_batch
        model_path = get_local_path(self.backend_opts.model_uri, tmp_dir)

        pretrained_uri = self.backend_opts.pretrained_uri
        if pretrained_uri:
            print('Loading weights from pretrained_uri: {}'.format(
                pretrained_uri))
            pretrained_path = download_if_needed(pretrained_uri, tmp_dir)
            learn.load(pretrained_path[:-4])

        # TODO add some validation metric and monitor it for doing the export
        callbacks = [
            TrackEpochCallback(learn),
            MySaveModelCallback(learn, every='epoch'),
            MyCSVLogger(learn, filename='log'),
            ExportCallback(learn, model_path, monitor='train_loss'),
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
        if self.model is None:
            self.print_options()
            model_uri = self.backend_opts.model_uri
            model_path = download_if_needed(model_uri, tmp_dir)
            checkpoint = torch.load(model_path, map_location='cpu')
            # add 2 for background and bogus classes
            num_classes = len(self.task_config.class_map) + 2
            model = fasterrcnn_resnet50_fpn(
                pretrained=False, progress=True, num_classes=num_classes,
                pretrained_backbone=True, min_size=300, max_size=300)
            model.load_state_dict(checkpoint['model'])
            model = model.to(self.device)
            self.model = model.eval()

    def predict(self, chips, windows, tmp_dir):
        """Return predictions for a chip using model.

        Args:
            chips: [[height, width, channels], ...] numpy array of chips
            windows: List of boxes that are the windows aligned with the chips.

        Return:
            Labels object containing predictions
        """
        self.load_model(tmp_dir)
        labels = ObjectDetectionLabels.make_empty()
        chips = torch.Tensor(chips).permute((0, 3, 1, 2)) / 255.
        chips = chips.to(self.device)
        output = self.model(chips)

        bogus_id = len(self.task_config.class_map) + 1

        for chip_ind, (chip, window) in enumerate(zip(chips, windows)):
            chip_output = output[chip_ind]
            boxes = chip_output['boxes'].detach().cpu().numpy().astype(np.float)
            class_ids = chip_output['labels'].detach().cpu().numpy()
            scores = chip_output['scores'].detach().cpu().numpy()

            # Remove any bogus boxes.
            good_inds = class_ids != bogus_id
            boxes = boxes[good_inds]
            class_ids = class_ids[good_inds]
            scores = scores[good_inds]

            # convert from (xmin, ymin, xmax, ymax) to (ymin, xmin, ymax, xmax)
            boxes = np.hstack((
                boxes[:, 1:2], boxes[:, 0:1], boxes[:, 3:4], boxes[:, 2:3]))
            boxes = ObjectDetectionLabels.local_to_global(
                boxes, window)
            class_ids = class_ids.astype(np.int32)

            labels = (labels + ObjectDetectionLabels(
                boxes, class_ids, scores=scores))

        return labels
