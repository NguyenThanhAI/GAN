import os
import argparse
from tqdm import tqdm

import cv2

import tensorflow.compat.v1 as tf


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _image_to_tfexample(image_data):
    return tf.train.Example(features=tf.train.Features(feature={"image": _bytes_feature(image_data)}))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--images_dir", type=str, default=r"D:\Anime_Face_Dataset\data")
    parser.add_argument("--saved_dir", type=str, default=r"D:\Anime_Face_Dataset\tfrecord")
    parser.add_argument("--tfrecord_output_file", type=str, default="anime_face_dataset.tfrecord")
    parser.add_argument("--image_size", type=int, default=80)

    args = parser.parse_args()

    return args


def enumerate_images(images_dir):
    images_list = []
    for dirs, _, files in os.walk(images_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                images_list.append(os.path.join(dirs, file))

    return images_list


if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.saved_dir):
        os.makedirs(args.saved_dir, exist_ok=True)

    images_list = enumerate_images(args.images_dir)

    image_phl = tf.placeholder(shape=[None, None, None], dtype=tf.uint8)

    jpg_encoded = tf.image.encode_jpeg(image_phl, quality=100)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
    config = tf.ConfigProto(gpu_options=gpu_options)

    num_corrupt_images = 0
    with tf.Session(config=config) as sess:
        with tf.python_io.TFRecordWriter(os.path.join(args.saved_dir, args.tfrecord_output_file)) as tfrecord_writer:
            for image in tqdm(images_list):
                img = cv2.imread(image)
                img = cv2.resize(img, (args.image_size, args.image_size))
                try:
                    img = img[:, :, ::-1]
                except:
                    num_corrupt_images += 1
                    print(image)
                    continue

                jpg_string = sess.run(jpg_encoded, feed_dict={image_phl: img})

                example = _image_to_tfexample(image_data=jpg_string)

                tfrecord_writer.write(example.SerializeToString())

            print("Num corrupt images: {}".format(num_corrupt_images))
