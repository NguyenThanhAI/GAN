import tensorflow.compat.v1 as tf
from tensorflow.contrib import slim

from ops import fully_connected, conv, deconv


def generator(inputs, dropout_rate=None, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
              is_training=True, reuse=None, scope=None):
    with tf.variable_scope(scope, "generator", [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + "_end_points"

        with slim.arg_scope([slim.dropout, slim.batch_norm], is_training=is_training), \
            slim.arg_scope([slim.conv2d, slim.conv2d_transpose, fully_connected, conv, deconv],
                           outputs_collections=end_points_collection), \
            slim.arg_scope([fully_connected, conv, deconv],
                           dropout_rate=dropout_rate, activation_fn=activation_fn), \
            slim.arg_scope([fully_connected, conv, deconv], normalizer_fn=normalizer_fn):

            net = inputs

            net = fully_connected(inputs=net, num_outputs=2 * 2 * 1024, scope="fully_connected_1")

            net = tf.reshape(net, shape=[-1, 2, 2, 1024], name="reshape")

            net = deconv(inputs=net, num_filters=512, kernel_size=5, stride=2, scope="deconv_1")

            net = deconv(inputs=net, num_filters=256, kernel_size=5, stride=2, scope="deconv_2")

            net = deconv(inputs=net, num_filters=128, kernel_size=5, stride=2, scope="deconv_3")

            net = deconv(inputs=net, num_filters=64, kernel_size=5, stride=2, scope="deconv_4")

            net = deconv(inputs=net, num_filters=3, kernel_size=5, stride=2, scope="deconv_5",
                         normalizer_fn=None, activation_fn=tf.nn.tanh)

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

        return net, end_points


def generator_arg_scope(weight_decay=1e-4, batch_norm_decay=0.99, batch_norm_epsilon=1.1e-5,
                        instance_norm_epsilon=1.1e-5, stddev_init=0.01):
    with slim.arg_scope([slim.fully_connected, slim.conv2d, slim.conv2d_transpose],
                        activation_fn=None):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            weights_regularizer=slim.l2_regularizer(scale=weight_decay),
                            biases_initializer=tf.zeros_initializer(),
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev_init)):
            with slim.arg_scope([slim.batch_norm],
                                scale=True,
                                decay=batch_norm_decay,
                                epsilon=batch_norm_epsilon):
                with slim.arg_scope([slim.instance_norm],
                                    scale=True,
                                    epsilon=instance_norm_epsilon) as scope:
                    return scope


def discriminator(inputs, dropout_rate=None, activation_fn=tf.nn.leaky_relu, normalizer_fn=slim.layer_norm,
                  is_training=True, reuse=None, scope=None):
    with tf.variable_scope(scope, "discriminator", [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + "_end_points"

        with slim.arg_scope([slim.dropout, slim.batch_norm], is_training=is_training), \
            slim.arg_scope([slim.conv2d, slim.conv2d_transpose, fully_connected, conv, deconv],
                           outputs_collections=end_points_collection), \
            slim.arg_scope([conv, deconv],
                           dropout_rate=dropout_rate, activation_fn=activation_fn), \
            slim.arg_scope([conv, deconv], normalizer_fn=normalizer_fn):

            net = inputs

            net = conv(inputs=net, num_filters=64, kernel_size=5, stride=2, scope="conv_1", normalizer_fn=None)

            net = conv(inputs=net, num_filters=128, kernel_size=5, stride=2, scope="conv_2")

            net = conv(inputs=net, num_filters=256, kernel_size=5, stride=2, scope="conv_3")

            net = conv(inputs=net, num_filters=512, kernel_size=5, stride=2, scope="conv_4")

            net = slim.flatten(net)

            net = fully_connected(inputs=net, num_outputs=1, scope="fully_connected_1", activation_fn=None,
                                  normalizer_fn=None)

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

        return net, end_points


def discriminator_arg_scope(weight_decay=1e-4, batch_norm_decay=0.99, batch_norm_epsilon=1.1e-5,
                            instance_norm_epsilon=1.1e-5, stddev_init=0.01):
    with slim.arg_scope([slim.fully_connected, slim.conv2d, slim.conv2d_transpose],
                        activation_fn=None):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            weights_regularizer=slim.l2_regularizer(scale=weight_decay),
                            biases_initializer=tf.zeros_initializer(),
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev_init)):
            with slim.arg_scope([slim.batch_norm],
                                scale=True,
                                decay=batch_norm_decay,
                                epsilon=batch_norm_epsilon):
                with slim.arg_scope([slim.instance_norm],
                                    scale=True,
                                    epsilon=instance_norm_epsilon) as scope:
                    return scope
