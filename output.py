import argparse
import tensorflow as tf
import numpy as np

from open3d_util import *
from models.pcn import PCN
from dataloader import Dataloader

def get_output(args):
    ds_valid = Dataloader(complete_dir=args.data_path, is_training=False, batch_size=1)
    ds_valid_iter = iter(ds_valid)

    # optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    model = PCN()
    # model.compile(optimizer=optimizer)
    # latest = tf.train.latest_checkpoint(args.checkpoint_dir)
    # print(latest)
    load_status = model.load_weights(args.checkpoint_dir).expect_partial()
    # load_status.assert_consumed()


    for step, (partial, npts, complete) in enumerate(ds_valid_iter):
        coarse, fine = model((partial, npts, complete), training=False)
        show_pcds([partial[0][0], fine[0]])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="data/test/complete")
    parser.add_argument('--checkpoint_dir', default="trained_model/cp-000093.ckpt")
    parser.add_argument('--save_output', default="/output/")
    args = parser.parse_args()

    get_output(args)