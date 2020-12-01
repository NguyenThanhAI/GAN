import cv2
import tensorflow.compat.v1 as tf


keys_to_features = {"image": tf.FixedLenFeature([], tf.string)}


def _parse_fn(data_record, in_size=80, out_size=64):
    features = keys_to_features
    sample = tf.parse_single_example(data_record, features)

    image = tf.image.decode_jpeg(sample["image"])
    image = tf.cast(image, dtype=tf.float32)
    image.set_shape(shape=[in_size, in_size, 3])
    image = tf.image.random_crop(image, size=[out_size, out_size, 3])
    image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_brightness(image, max_delta=5.)
    #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    #image = tf.image.random_hue(image, max_delta=0.2)
    #image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.divide(tf.subtract(image, 127.5 * tf.ones_like(image)), 127.5 * tf.ones_like(image))

    return image


def get_batch(tfrecord_path, batch_size, num_epochs=100):
    dataset = tf.data.TFRecordDataset([tfrecord_path])
    dataset = dataset.map(_parse_fn)
    dataset = dataset.shuffle(2000)
    dataset = dataset.batch(batch_size)
    epoch = tf.data.Dataset.range(num_epochs)
    dataset = epoch.flat_map(lambda i: tf.data.Dataset.zip((dataset, tf.data.Dataset.from_tensors(i).repeat())))
    dataset = dataset.repeat(num_epochs)

    iterator = dataset.make_one_shot_iterator()
    (image), epoch = iterator.get_next()

    return image, epoch


#image, epoch_now = get_batch(r"D:\Anime_Face_Dataset\tfrecord\anime_face_dataset.tfrecord", 1, 1)
#
#image = tf.cast(tf.add(tf.multiply(image, 127.5 * tf.ones_like(image)), 127.5 * tf.ones_like(image)), dtype=tf.uint8)
#
#with tf.Session() as sess:
#    try:
#        while True:
#            img = sess.run(image)
#
#            imag = img[0][:, :, ::-1]
#
#            cv2.imshow("Image", imag)
#            cv2.waitKey(100)
#    except tf.errors.OutOfRangeError:
#        print("Finished")
#        pass#