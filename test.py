import tensorflow as tf
import argparse
import os

from open3d_util import *
from datagenerator import DataGenerator
from models.pcn import PCN

def train(args):
    ds_test = DataGenerator(os.listdir(args.test_dir), complete_dir=args.data_path, batch_size=args.batch_size)

    # Model Initialization
    model = PCN()
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001)
    model.compile(optimizer=optimizer)
    latest = tf.train.latest_checkpoint(args.checkpoint_dir)
    model.load_weights(latest)

    # Testing
    for id, (input, npts, gt) in enumerate(ds_test):
        coarse, fine = model((input, npts), training=True)
        if args.visualise_outputs:
            show_pcds([input[0], fine[0], gt[0]])

        if args.save_outputs:
            #TODO
            pass
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', default="/content/drive/MyDrive/")
    parser.add_argument('--checkpoint_dir', default="/content/drive/MyDrive/pcn_tf_2/")
    parser.add_argument('--visualise_outputs', action='store_true')
    parser.add_argument('--save_outputs', action='store_true')

    args = parser.parse_args()

    train(args)