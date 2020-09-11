import numpy as np

import tensorflow as tf
from tensorflow.contrib import slim

from models import generator_arg_scope, generator, discriminator_arg_scope, discriminator

tf.set_random_seed(1000)

noise = tf.random.uniform(shape=[10, 128], maxval=1.)

with slim.arg_scope(generator_arg_scope()):
    fake_images, end_points = generator(inputs=noise, scope="generator")

print(fake_images, end_points)

trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
num_vars = 0
for vars in trainable_variables:
    num_vars += np.prod(vars.shape)

print(num_vars)

with slim.arg_scope(discriminator_arg_scope()):
    fake_prob, end_points = discriminator(inputs=fake_images, scope="discriminator", reuse=False)
    fake_prob_1, end_points_1 = discriminator(inputs=fake_images, scope="discriminator", reuse=True)

gradients = tf.gradients(fake_prob, [fake_images])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
print(slopes)
gradient_penalty = tf.reduce_mean(tf.square(slopes - tf.ones_like(slopes)))
print(gradient_penalty)

trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
num_vars = 0
for vars in trainable_variables:
    num_vars += np.prod(vars.shape)

print(num_vars)

trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
num_vars = 0
for vars in trainable_variables:
    num_vars += np.prod(vars.shape)

print(num_vars)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    fk_pr, fk_pr_1 = sess.run([fake_prob, fake_prob_1])

    print(fk_pr, fk_pr_1)

    sl, gp = sess.run([slopes, gradient_penalty])
    print(sl, gp)
