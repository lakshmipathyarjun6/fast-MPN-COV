import argparse
import os
import shutil

import tensorflow as tf
import glob
import math

from PIL import Image
from PIL import ImageDraw

import random

IMAGE_FEATURE_DESCRIPTION = {
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/source_id': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
}

def generateTrainingData(dataset, top_folder):
    train_dir = os.path.join(top_folder, 'train')
    val_dir = os.path.join(top_folder, 'val')
    tmp_dir = os.path.join(top_folder, 'tmp')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    item = 0

    for value in dataset:
        parsed = tf.io.parse_single_example(value, IMAGE_FEATURE_DESCRIPTION)

        tf_image = tf.image.decode_jpeg(parsed['image/encoded'])
        image = Image.fromarray(tf_image.numpy())
        tf_label = parsed['image/object/class/text'].values.numpy()[0].decode("utf-8")

        tmp_path = os.path.join(tmp_dir, tf_label)

        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        savepath = os.path.join(tmp_path, '{}.jpg'.format(item))
        image.save(savepath)

        print('Adding image ', item)
        item += 1

    dirs = glob.glob(tmp_dir + "/*")

    for dir in dirs:
        label = dir.split("/")[-1]

        items = glob.glob(dir + "/*")
        numItems = len(items)

        train_cutoff = math.floor(numItems * 0.8)

        random.shuffle(items)

        train_set = items[:train_cutoff]
        val_set = items[train_cutoff:]

        train_path = os.path.join(train_dir, label)

        if not os.path.exists(train_path):
            os.makedirs(train_path)

        val_path = os.path.join(val_dir, label)

        if not os.path.exists(val_path):
            os.makedirs(val_path)

        tset_pad = len(str(len(train_set)))
        vset_pad = len(str(len(val_set)))

        train_index = 1
        val_index = 1

        for item in train_set:
            newpath = os.path.join(train_path, '{}_{}.jpg'.format(label, str(train_index).zfill(tset_pad)))
            shutil.move(item, newpath)
            train_index += 1

        for item in val_set:
            newpath = os.path.join(val_path, '{}_{}.jpg'.format(label, str(val_index).zfill(vset_pad)))
            shutil.move(item, newpath)
            val_index += 1

    shutil.rmtree(tmp_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', nargs='?', required=True)
    args = parser.parse_args()

    source = args.source
    top_folder = source.split('.tfrecord')[0]

    if not os.path.exists(top_folder):
        os.makedirs(top_folder)

    dataset = tf.data.TFRecordDataset(source)

    generateTrainingData(dataset, top_folder)
    
if __name__ == '__main__':
    main()
