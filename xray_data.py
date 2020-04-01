import os
import random
import xml.etree.ElementTree as ET
from functools import partial
from glob import glob

import numpy as np
import tensorflow as tf
from PIL import Image

from box_utils import compute_target
from image_utils import horizontal_flip, random_patching


class XRayDataset():
    """ Class for XRayDataset

    Attributes:
        root_dir: dataset root dir (ex: ./data/VOCdevkit)
        num_examples: number of examples to be used
                      (in case one wants to overfit small data)
    """

    def __init__(self, root_dir, default_boxes,
                 new_size, num_examples=-1, augmentation=None):
        super(XRayDataset, self).__init__()
        self.idx_to_name = [
            'sponge1', 'needle']
        self.name_to_idx = dict([(v, k)
                                 for k, v in enumerate(self.idx_to_name)])

        self.data_dir = root_dir
        self.ids = list(map(lambda x: os.path.basename(x).split('.')[0], glob(os.path.join(self.data_dir, '*implant*.xml'))))
        random.shuffle(self.ids)
        self.default_boxes = default_boxes
        self.new_size = new_size

        if num_examples != -1:
            self.ids = self.ids[:num_examples]

        self.train_ids = self.ids[:int(len(self.ids) * 0.75)]
        self.val_ids = self.ids[int(len(self.ids) * 0.75):]

        if augmentation == None:
            self.augmentation = ['original']
        else:
            self.augmentation = augmentation + ['original']

    def __len__(self):
        return len(self.ids)

    def _get_image(self, index):
        """ Method to read image from file
            then resize to (300, 300)
            then subtract by ImageNet's mean
            then convert to Tensor

        Args:
            index: the index to get filename from self.ids

        Returns:
            img: tensor of shape (3, 300, 300)
        """
        filename = self.ids[index]
        img_path = os.path.join(self.data_dir, filename + '.implant.png')
        img = Image.open(img_path)

        return img

    def _get_annotation(self, index, orig_shape):
        """ Method to read annotation from file
            Boxes are normalized to image size
            Integer labels are increased by 1

        Args:
            index: the index to get filename from self.ids
            orig_shape: image's original shape

        Returns:
            boxes: numpy array of shape (num_gt, 4)
            labels: numpy array of shape (num_gt,)
        """
        h, w = orig_shape
        filename = self.ids[index]
        anno_path = os.path.join(self.data_dir, filename + '.implant.sponge1.posWin.xml')
        windows = ET.parse(anno_path).find('Windows') # Get the Windows tag
        objects = windows.findall('_') # Find all objects in image
        boxes = []
        labels = []

        for obj in objects:
            name = 'sponge1'
            # name = obj.find('name').text.lower().strip()
            xmin = (float(obj.find('x').text))
            ymin = (float(obj.find('y').text))
            
            xmin = (float(obj.find('x').text) - 1) / w
            ymin = (float(obj.find('y').text) - 1) / h
            xmax = (float(obj.find('width').text) - 1) / w + xmin
            ymax = (float(obj.find('height').text) - 1) / h + ymin
            boxes.append([xmin, ymin, xmax, ymax])

            labels.append(self.name_to_idx[name] + 1)

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def generate(self, subset=None):
        """ The __getitem__ method
            so that the object can be iterable

        Args:
            index: the index to get filename from self.ids

        Returns:
            img: tensor of shape (300, 300, 3)
            boxes: tensor of shape (num_gt, 4)
            labels: tensor of shape (num_gt,)
        """
        if subset == 'train':
            indices = self.train_ids
        elif subset == 'val':
            indices = self.val_ids
        else:
            indices = self.ids
        for index in range(len(indices)):
            # img, orig_shape = self._get_image(index)
            filename = indices[index]
            img = self._get_image(index)
            w, h = img.size
            boxes, labels = self._get_annotation(index, (h, w))
            boxes = tf.constant(boxes, dtype=tf.float32)
            labels = tf.constant(labels, dtype=tf.int64)

            augmentation_method = np.random.choice(self.augmentation)
            if augmentation_method == 'patch':
                img, boxes, labels = random_patching(img, boxes, labels)
            elif augmentation_method == 'flip':
                img, boxes, labels = horizontal_flip(img, boxes, labels)

            img = np.array(img.resize(
                (self.new_size, self.new_size)), dtype=np.float32)
            img = (img / 127.0) - 1.0
            img = tf.constant(img, dtype=tf.float32)

            gt_confs, gt_locs = compute_target(
                self.default_boxes, boxes, labels)

            yield filename, img, gt_confs, gt_locs


def create_batch_generator(root_dir, default_boxes,
                           new_size, batch_size, num_batches,
                           mode,
                           augmentation=None):
    num_examples = batch_size * num_batches if num_batches > 0 else -1
    xray = XRayDataset(root_dir, default_boxes,
                       new_size, num_examples, augmentation)

    info = {
        'idx_to_name': xray.idx_to_name,
        'name_to_idx': xray.name_to_idx,
        'length': len(xray),
        'image_dir': xray.data_dir,
    }

    if mode == 'train':
        train_gen = partial(xray.generate, subset='train')
        train_dataset = tf.data.Dataset.from_generator(
            train_gen, (tf.string, tf.float32, tf.int64, tf.float32))
        val_gen = partial(xray.generate, subset='val')
        val_dataset = tf.data.Dataset.from_generator(
            val_gen, (tf.string, tf.float32, tf.int64, tf.float32))

        train_dataset = train_dataset.shuffle(40).batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)

        return train_dataset.take(num_batches), val_dataset.take(-1), info
    else:
        dataset = tf.data.Dataset.from_generator(
            xray.generate, (tf.string, tf.float32, tf.int64, tf.float32))
        dataset = dataset.batch(batch_size)
        return dataset.take(num_batches), info



from anchor import generate_default_boxes
import yaml
if __name__ == "__main__":

    with open('./config.yml') as f:
        cfg = yaml.safe_load(f)

    try:
        config = cfg['SSD300']
    except AttributeError:
        raise ValueError('Unknown architecture: {}'.format(config))

    default_boxes = generate_default_boxes(config)

    batch_generator, val_generator, info = create_batch_generator(
        '../xraydata/sponge1/', 
        default_boxes,
        300,
        32,
        -1, 
        'train'
    )

    for i, (_, imgs, gt_confs, gt_locs) in enumerate(batch_generator):
        print(f'Image {i}, {gt_confs}')