import tensorflow as tf
import argparse
import os

from datagenerator import DataGenerator
from models.pcn import PCN

def train(args):
    # Data Pre-paration
    ds_train = DataGenerator(complete_dir=args.data_path, is_training=True, batch_size=args.batch_size)
    ds_valid = DataGenerator(complete_dir=args.data_path, is_training=False, batch_size=args.batch_size)

    # Model Initialization
    if args.restore:
        latest = tf.train.latest_checkpoint(args.checkpoint_dir)
        model = PCN()
        model.load_weights(latest)
    else:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args.base_lr,
        decay_steps=args.decay_steps,
        decay_rate=args.decay_rate,
        staircase=True)
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model = PCN()
        model.compile(optimizer=optimizer)
    

    
    print("Begin Training".center(100,"-"))
    model.fit(ds_train, validation_data=ds_valid)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="data/complete/")
    parser.add_argument('--checkpoint_dir', default="/content/drive/MyDrive/pcn_tf_2/")
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--decay_steps', type=int, default=30000)
    parser.add_argument('--decay_rate', type=float, default=0.8)
    parser.add_argument('--log_freq', type=float, default=100)
    parser.add_argument('--eval_freq', type=float, default=2)
    parser.add_argument('--save_freq', type=float, default=5)
    args = parser.parse_args()

    train(args)