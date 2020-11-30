import sys
import os
import argparse

ROOT_DIR = os.path.abspath("..")
sys.path.append(ROOT_DIR)

from WGAN.wgan_gp import WGANGPConfig, WGANGP


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tfrecord_path", type=str, default=r"D:\Anime_Face_Dataset\tfrecord\anime_face_dataset.tfrecord")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--is_training", type=str2bool, default=True)
    parser.add_argument("--noise_type", type=str, default="unsigned_uniform")
    parser.add_argument("--dropout_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_a", type=float, default=10.)
    parser.add_argument("--n_critic", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_rate_decay_type", type=str, default="constant")
    parser.add_argument("--decay_steps", type=int, default=10000)
    parser.add_argument("--decay_rate", type=float, default=0.9)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta_1", type=float, default=0.5)
    parser.add_argument("--beta_2", type=float, default=0.9)
    parser.add_argument("--per_process_gpu_memory_fraction", type=float, default=0.95)
    parser.add_argument("--is_loadmodel", type=str2bool, default=False)
    parser.add_argument("--summary_dir", type=str, default="summary")
    parser.add_argument("--generator_model_dir", type=str, default="generator_saved_model")
    parser.add_argument("--generator_checkpoint_name", type=str, default=None)
    parser.add_argument("--discriminator_model_dir", type=str, default="discriminator_saved_model")
    parser.add_argument("--discriminator_checkpoint_name", type=str, default=None)
    parser.add_argument("--summary_frequency", type=int, default=10)
    parser.add_argument("--save_network_frequency", type=int, default=10000)
    parser.add_argument("--debug_mode", type=str2bool, default=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    print("Args: {}".format(args))

    if not os.path.exists(args.summary_dir):
        os.makedirs(args.summary_dir, exist_ok=True)

    if not os.path.exists(args.generator_model_dir):
        os.makedirs(args.generator_model_dir, exist_ok=True)

    if not os.path.exists(args.discriminator_model_dir):
        os.makedirs(args.discriminator_model_dir, exist_ok=True)

    config = WGANGPConfig(tfrecord_path=args.tfrecord_path,
                          batch_size=args.batch_size,
                          num_epochs=args.num_epochs,
                          z_dim=args.z_dim,
                          is_training=args.is_training,
                          noise_type=args.noise_type,
                          dropout_rate=args.dropout_rate,
                          weight_decay=args.weight_decay,
                          lambda_a=args.lambda_a,
                          n_critic=args.n_critic,
                          learning_rate=args.learning_rate,
                          learning_rate_decay_type=args.learning_rate_decay_type,
                          decay_steps=args.decay_steps,
                          decay_rate=args.decay_rate,
                          optimizer=args.optimizer,
                          momentum=args.momentum,
                          beta_1=args.beta_1,
                          beta_2=args.beta_2,
                          per_process_gpu_memory_fraction=args.per_process_gpu_memory_fraction,
                          is_loadmodel=args.is_loadmodel,
                          summary_dir=args.summary_dir,
                          generator_model_dir=args.generator_model_dir,
                          generator_checkpoint_name=args.generator_checkpoint_name,
                          discriminator_model_dir=args.discriminator_model_dir,
                          discriminator_checkpoint_name=args.discriminator_checkpoint_name,
                          summary_frequency=args.summary_frequency,
                          save_network_frequency=args.save_network_frequency,
                          debug_mode=args.debug_mode)

    wgan_gp = WGANGP(config=config)

    wgan_gp.train()
