import tensorflow as tf
import argparse
import os

from datagenerator import DataGenerator
from models.pcn import PCN

def get_file_paths(complete_dir, is_training=True):
    mid_pt = int(len(os.listdir(complete_dir))*7/9)
    start = 0 if is_training else mid_pt
    end = mid_pt if is_training else None
    return sorted(os.listdir(complete_dir))[start:end]

def train(args):
    ds_train = DataGenerator(get_file_paths(args.data_path, is_training=True),complete_dir=args.data_path, batch_size=args.batch_size)
    ds_valid = DataGenerator(get_file_paths(args.data_path, is_training=False),complete_dir=args.data_path, batch_size=args.batch_size)

    # Model Initialization
    model = PCN()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    args.base_lr,
    decay_steps=args.decay_steps,
    decay_rate=args.decay_rate,
    staircase=True)
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer)
    if args.restore:
        latest = tf.train.latest_checkpoint(args.restore_point)
        model.load_weights(latest)
    
    print("Begin Training".center(100,"-"))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, verbose=1)
    checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath=args.checkpoint_dir, save_weights_only=True,
                                            monitor='val_loss', mode='min', verbose=1,save_best_only=True)
    hist = model.fit(ds_train, validation_data=ds_valid, epochs=args.num_epochs, callbacks=[early_stopping, checkpoints], workers=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="data/complete/")
    parser.add_argument('--checkpoint_dir', default="/content/drive/MyDrive/pcn_tf_2/{epoch:003d}")
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--restore_point', default="/content/drive/MyDrive/pcn_tf_2/")

    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--decay_steps', type=int, default=30000)
    parser.add_argument('--decay_rate', type=float, default=0.8)
    parser.add_argument('--save_freq', type=int, default=5)
    args = parser.parse_args()

    train(args)