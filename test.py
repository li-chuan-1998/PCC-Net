import tensorflow as tf
import argparse
import os

from open3d_util import *
from datagenerator import DataGenerator
from models.pcn import PCN

def train(args):
    id_list = os.listdir(args.test_dir)
    ds_test = DataGenerator(id_list, complete_dir=args.test_dir, batch_size=args.batch_size, shuffle=False)

    # Model Initialization
    model = PCN()
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001)
    model.compile(optimizer=optimizer)
    latest = tf.train.latest_checkpoint(args.checkpoint_dir)
    model.load_weights(latest)

    # Testing
    base = 0.01
    for id, (inputs, gt) in enumerate(ds_test):
        coarse, fine = model(inputs, training=False)
        if args.visualise_outputs:
            show_pcds([inputs[0][0], fine[0] + base, gt[0]+base*2])

        if args.save_outputs:
            filename = id_list[id].replace("complete", "partial")
            save_pcd(os.path.join(args.save_dir, filename), fine)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', default="/content/drive/MyDrive/")
    parser.add_argument('--checkpoint_dir', default="/content/drive/MyDrive/pcn_tf_2/")
    parser.add_argument('--visualise_outputs', action='store_true')
    parser.add_argument('--save_outputs', action='store_true')
    parser.add_argument('--save_dir', default="/content/drive/MyDrive/")
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()

    train(args)