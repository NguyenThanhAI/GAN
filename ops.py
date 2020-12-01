import tensorflow.compat.v1 as tf
from tensorflow.contrib import slim


@slim.add_arg_scope
def fully_connected(inputs, num_outputs, dropout_rate=None, scope=None,
                    outputs_collections=None, activation_fn=tf.nn.relu):
    with tf.variable_scope(scope, "fully_connected", [inputs]) as sc:
        net = slim.fully_connected(inputs=inputs, num_outputs=num_outputs)

        if activation_fn is not None:
            net = activation_fn(net)

        if dropout_rate:
            net = slim.dropout(net, keep_prob=dropout_rate)

        net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net


@slim.add_arg_scope
def conv(inputs, num_filters, kernel_size, stride=1,
         dropout_rate=None, scope=None, outputs_collections=None,
         activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
    with tf.variable_scope(scope, "conv", [inputs]) as sc:
        net = slim.conv2d(inputs=inputs, num_outputs=num_filters, kernel_size=kernel_size, stride=stride)
        if normalizer_fn is not None:
            net = normalizer_fn(net)

        if activation_fn is not None:
            net = activation_fn(net)

        if dropout_rate:
            net = slim.dropout(net, keep_prob=dropout_rate)

        net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net


@slim.add_arg_scope
def deconv(inputs, num_filters, kernel_size=4, stride=2,
           dropout_rate=None, scope=None, outputs_collections=None,
           activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
    with tf.variable_scope(scope, "deconv", [inputs]) as sc:
        net = slim.conv2d_transpose(inputs=inputs, num_outputs=num_filters, kernel_size=kernel_size, stride=stride)
        if normalizer_fn is not None:
            net = normalizer_fn(net)

        if activation_fn is not None:
            net = activation_fn(net)

        if dropout_rate:
            net = slim.dropout(net, keep_prob=dropout_rate)

        net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net
