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
    optimizer=tf.keras.optimizers.Adam(learning_rate=args.base_lr)
    model.compile(optimizer=optimizer)
    
    # Callbacks
    resume_training = tf.keras.callbacks.BackupAndRestore(args.resume_training_dir)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1)
    checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath=args.checkpoint_dir, save_weights_only=True,
                                            monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=args.decay_rate, patience=2, min_lr=0.000001)
    callbacks = [resume_training, early_stopping, checkpoints, reduce_lr]

    # Training/Validating
    hist = model.fit(ds_train, validation_data=ds_valid, epochs=args.num_epochs, callbacks=callbacks, workers=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="data/complete/")
    parser.add_argument('--checkpoint_dir', default="/content/drive/MyDrive/pcn_tf_2/{epoch:003d}-({val_loss:.6f})")
    parser.add_argument('--resume_training_dir', default="/content/drive/MyDrive/pcn_tf_2/backup/")

    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--decay_rate', type=float, default=0.3)
    args = parser.parse_args()

    train(args)