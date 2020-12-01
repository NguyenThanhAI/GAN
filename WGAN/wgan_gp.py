import os
from datetime import datetime
import time

import tensorflow.compat.v1 as tf
from tensorflow.python import debug as tf_debug
from tensorflow.contrib import slim

from models import generator_arg_scope, generator, discriminator_arg_scope, discriminator
from read_tfrecord import get_batch


class WGANGPConfig(object):
    def __init__(self, tfrecord_path=r"D:\Anime_Face_Dataset\tfrecord\anime_face_dataset.tfrecord",
                 batch_size=64, num_epochs=2000, z_dim=128, is_training=True, noise_type="unsigned_uniform",
                 dropout_rate=None, weight_decay=1e-4, lambda_a=10., n_critic=5, learning_rate=1e-4,
                 learning_rate_decay_type="constant", decay_steps=10000, decay_rate=0.9, optimizer="adam", momentum=0.9,
                 beta_1=0.5, beta_2=0.9, per_process_gpu_memory_fraction=0.95, is_loadmodel=False,
                 summary_dir="./summary", generator_model_dir="./generator_saved_model", generator_checkpoint_name=None,
                 discriminator_model_dir="discriminator_saved_model",
                 discriminator_checkpoint_name=None, summary_frequency=10, save_network_frequency=10000,
                 debug_mode=False):
        self.tfrecord_path = tfrecord_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.z_dim = z_dim
        self.is_training = is_training
        self.noise_type = noise_type
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.lambda_a = lambda_a
        self.n_critic = n_critic
        self.learning_rate = learning_rate
        self.learning_rate_decay_type = learning_rate_decay_type
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.momentum = momentum
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
        self.is_loadmodel = is_loadmodel
        self.summary_dir = summary_dir
        self.generator_model_dir = generator_model_dir
        self.generator_checkpoint_name = generator_checkpoint_name
        self.discriminator_model_dir = discriminator_model_dir
        self.discriminator_checkpoint_name = discriminator_checkpoint_name
        self.summary_frequency = summary_frequency
        self.save_network_frequency = save_network_frequency
        self.debug_mode = debug_mode



class WGANGP(object):
    def __init__(self, config: WGANGPConfig):
        self.config = config
        tf.reset_default_graph()

        self.real_images, self.epoch_now = get_batch(tfrecord_path=self.config.tfrecord_path,
                                                     batch_size=self.config.batch_size,
                                                     num_epochs=self.config.num_epochs)

        self.noise_z = self.generate_noise()

        with slim.arg_scope(generator_arg_scope()):
            self.fake_images, _ = generator(inputs=self.noise_z, dropout_rate=self.config.dropout_rate,
                                            is_training=self.config.is_training, reuse=False, scope="generator")

        with slim.arg_scope(discriminator_arg_scope()):
            self.fake_logits, _ = discriminator(inputs=self.fake_images, dropout_rate=self.config.dropout_rate,
                                                is_training=self.config.is_training, reuse=False, scope="discriminator")
            self.real_logits, _ = discriminator(inputs=self.real_images, dropout_rate=self.config.dropout_rate,
                                                is_training=self.config.is_training, reuse=True, scope="discriminator")

        self.global_step = tf.train.get_or_create_global_step()
        self.global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + \
            tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)

        self.generator_trainable_variables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                               scope="generator")
        self.generator_saver = tf.train.Saver(var_list=self.generator_trainable_variables + [self.global_step])

        self.discriminator_trainable_variables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                   scope="discriminator")
        self.discriminator_saver = tf.train.Saver(var_list=self.discriminator_trainable_variables)

        with tf.name_scope("generator_loss"):
            self.generator_regularization_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in self.generator_trainable_variables])
            tf.summary.scalar("regularization loss", self.generator_regularization_loss)
            self.generator_loss = - tf.reduce_mean(self.fake_logits)
            tf.summary.scalar("loss", self.generator_loss)
            self.total_generator_loss = self.generator_loss + self.config.weight_decay * self.generator_regularization_loss
            tf.summary.scalar("total loss", self.total_generator_loss)

        with tf.name_scope("discriminator_loss"):
            self.discriminator_regularization_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in self.discriminator_trainable_variables])
            tf.summary.scalar("regularization loss", self.discriminator_regularization_loss)
            self.discriminator_loss = tf.reduce_mean(self.fake_logits) - tf.reduce_mean(self.real_logits)
            tf.summary.scalar("loss", self.discriminator_loss)

            self.epsilon = tf.random.uniform(shape=[tf.shape(self.real_images)[0], 1, 1, 1], minval=0., maxval=1.)
            self.interpolates = self.epsilon * self.real_images + (1. - self.epsilon) * self.fake_images
            self.interpolates_logits, _ = discriminator(inputs=self.interpolates, dropout_rate=self.config.dropout_rate,
                                                        is_training=self.config.is_training,
                                                        reuse=True, scope="discriminator")
            self.gradients = tf.gradients(self.interpolates_logits, [self.interpolates])[0]
            self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), axis=[1, 2, 3]) + 1e-12)
            self.gradient_penalty = tf.reduce_mean(tf.square(self.slopes - tf.ones_like(self.slopes)))
            tf.summary.scalar("gradient penalty", self.gradient_penalty)
            self.total_discriminator_loss = self.discriminator_loss + self.config.lambda_a * self.gradient_penalty + self.config.weight_decay * self.discriminator_regularization_loss
            tf.summary.scalar("total loss", self.total_discriminator_loss)

        with tf.name_scope("image"):
            tf.summary.image("fake images", tf.cast(
                tf.add(tf.multiply(self.fake_images, 127.5 * tf.ones_like(self.fake_images)),
                       127.5 * tf.ones_like(self.fake_images)), dtype=tf.uint8))
            tf.summary.image("real images", tf.cast(
                tf.add(tf.multiply(self.real_images, 127.5 * tf.ones_like(self.real_images)),
                       127.5 * tf.ones_like(self.real_images)), dtype=tf.uint8))

        with tf.name_scope("learning_rate"):
            if self.config.learning_rate_decay_type == "constant":
                self.learning_rate = self.config.learning_rate
            elif self.config.learning_rate_decay_type == "piecewise_constant":
                self.learning_rate = tf.train.piecewise_constant(x=self.global_step,
                                                                 boundaries=[20000, 200000, 500000],
                                                                 values=[5e-4, 2.5e-4, 1e-4, 5e-5])
            elif self.config.learning_rate_decay_type == "exponential_decay":
                self.learning_rate = tf.train.exponential_decay(learning_rate=self.config.learning_rate,
                                                                global_step=self.global_step,
                                                                decay_steps=self.config.decay_steps,
                                                                decay_rate=self.config.decay_rate,
                                                                staircase=True)
            elif self.config.learning_rate_decay_type == "linear_cosine_decay":
                self.learning_rate = tf.train.linear_cosine_decay(learning_rate=self.config.learning_rate,
                                                                  global_step=self.global_step,
                                                                  decay_steps=self.config.decay_steps)
            else:
                raise ValueError("Learning rate decay type {} was not recognized".format(self.config.learning_rate_decay_type))
            tf.summary.scalar("learning rate", self.learning_rate)

        with tf.name_scope("optimizer"):
            if self.config.optimizer.lower() == "adam":
                self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                                  beta1=self.config.beta_1,
                                                                  beta2=self.config.beta_2)
                self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                                      beta1=self.config.beta_1,
                                                                      beta2=self.config.beta_2)
            elif self.config.optimizer.lower() == "rms" or self.config.optimizer.lower() == "rmsprop":
                self.generator_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                                                     momentum=self.config.momentum)
                self.discriminator_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                                                         momentum=self.config.momentum)
            elif self.config.optimizer.lower() == "momentum":
                self.generator_optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                                      momentum=self.config.momentum)
                self.discriminator_optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                                          momentum=self.config.momentum)
            elif self.config.optimizer.lower() == "grad_descent":
                self.generator_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
                self.discriminator_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            else:
                raise ValueError("Optimizer {} was not recognized".format(self.config.optimizer))

            self.generator_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="generator")
            with tf.control_dependencies(self.generator_update_ops):
                self.generator_train_op = self.generator_optimizer.minimize(loss=self.total_generator_loss,
                                                                            global_step=self.global_step,
                                                                            var_list=self.generator_trainable_variables)

            self.discriminator_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="discriminator")
            with tf.control_dependencies(self.discriminator_update_ops):
                self.discriminator_train_op = self.discriminator_optimizer.minimize(loss=self.total_discriminator_loss,
                                                                                    global_step=self.global_step,
                                                                                    var_list=self.discriminator_trainable_variables)

        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.config.per_process_gpu_memory_fraction,
                                         allow_growth=True)
        self.config_gpu = tf.ConfigProto(gpu_options=self.gpu_options,
                                         allow_soft_placement=True,
                                         log_device_placement=True)

        self.sess = tf.Session(config=self.config_gpu)
        if self.config.debug_mode:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(sess=self.sess)

        self.merged = tf.summary.merge_all()

        if self.config.is_loadmodel:
            self.writer = tf.summary.FileWriter(logdir=self.config.summary_dir)
        else:
            self.writer = tf.summary.FileWriter(logdir=self.config.summary_dir, graph=self.sess.graph)

        self.restore_or_initialize_network(generator_checkpoint_name=self.config.generator_checkpoint_name,
                                           discriminator_checkpoint_name=self.config.discriminator_checkpoint_name)

    def generate_noise(self):
        if self.config.is_training:
            if self.config.noise_type == "unsigned_uniform":
                noise_z = tf.random.uniform(shape=[self.config.batch_size, self.config.z_dim],
                                                 minval=0.,
                                                 maxval=1.,
                                                 name="noise_z")
            elif self.config.noise_type == "signed_uniform":
                noise_z = tf.random.uniform(shape=[self.config.batch_size, self.config.z_dim],
                                                 minval=-1.,
                                                 maxval=1.,
                                                 name="noise_z")
            elif self.config.noise_type == "normal":
                noise_z = tf.random.normal(shape=[self.config.batch_size, self.config.z_dim],
                                                mean=0.0,
                                                stddev=1.0,
                                                name="noise_z")
            else:
                raise ValueError("Random noise type {} was not recognized".format(self.config.noise_type))
        else:
            noise_z = tf.placeholder(dtype=tf.float32, shape=[None, self.config.z_dim], name="noise_z")

        return noise_z

    def restore_or_initialize_network(self, generator_checkpoint_name=None, discriminator_checkpoint_name=None):
        self.sess.run(tf.global_variables_initializer())
        if self.config.is_loadmodel:
            if generator_checkpoint_name is not None:
                self.generator_saver.restore(sess=self.sess,
                                             save_path=os.path.join(self.config.generator_model_dir, generator_checkpoint_name))
            else:
                self.generator_saver.restore(sess=self.sess,
                                             save_path=tf.train.latest_checkpoint(checkpoint_dir=self.config.generator_model_dir))
            if discriminator_checkpoint_name is not None:
                self.discriminator_saver.restore(sess=self.sess,
                                                 save_path=os.path.join(self.config.discriminator_model_dir, discriminator_checkpoint_name))
            else:
                self.discriminator_saver.restore(sess=self.sess,
                                                 save_path=tf.train.latest_checkpoint(checkpoint_dir=self.config.discriminator_model_dir))
            print("Successfully load model at {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        else:
            print("Successfully initialize model at {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

    def get_global_step(self):
        return tf.train.global_step(sess=self.sess, global_step_tensor=self.global_step)

    def save_networks(self, step):
        if not os.path.exists(self.config.generator_model_dir):
            os.makedirs(self.config.generator_model_dir, exist_ok=True)
        if not os.path.exists(self.config.discriminator_model_dir):
            os.makedirs(self.config.discriminator_model_dir, exist_ok=True)

        generator_checkpoint_name = "Generator-" + str(step).zfill(7)
        self.generator_saver.save(sess=self.sess, save_path=os.path.join(self.config.generator_model_dir, generator_checkpoint_name))

        discriminator_checkpoint_name = "Discriminator-" + str(step).zfill(7)
        self.discriminator_saver.save(sess=self.sess, save_path=os.path.join(self.config.discriminator_model_dir, discriminator_checkpoint_name))

        print("Save network at {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

    def add_summary(self, step):
        summary = self.sess.run(self.merged)

        self.writer.add_summary(summary=summary, global_step=step)
        self.writer.flush()

        print("Add summary at {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

    def train(self):
        step = self.get_global_step() // (self.config.n_critic + 1)
        try:
            while True:
                start = time.time()
                step += 1
                for i in range(self.config.n_critic):
                    _, dis_cost = self.sess.run([self.discriminator_train_op, self.total_discriminator_loss])
                    print("Update discriminator time {} of Step {}, discriminator loss: {},".format(i + 1, step, round(dis_cost, ndigits=3)), end=" ")
                _, gen_cost = self.sess.run([self.generator_train_op, self.total_generator_loss])

                if step % self.config.summary_frequency == 0:
                    self.add_summary(step=step)
                if step % self.config.save_network_frequency == 0:
                    self.save_networks(step=step)
                end = time.time()
                print("Step: {}, generator loss: {}, takes time: {}".format(step, round(gen_cost, ndigits=3), round(end - start, ndigits=3)))
        except tf.errors.OutOfRangeError:
            self.save_networks(step=step)
            print("Training process finished")
            pass
