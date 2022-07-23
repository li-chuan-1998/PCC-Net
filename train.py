import tensorflow as tf
import numpy as np
import argparse

from models.pcn import PCN
from loss_utils.utils import *
from dataloader import Dataloader

def train(args):
    # Data Pre-paration
    ds_train = Dataloader(complete_dir=args.data_path, is_training=True, batch_size=args.batch_size)
    ds_train_iter = iter(ds_train)
    # ds_valid = Dataloader(complete_dir="data/complete/", is_training=False, batch_size=8)

    # Training & Validation
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args.base_lr,
        decay_steps=args.decay_steps,
        decay_rate=args.decay_rate,
        staircase=True)
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model = PCN()
    
    for epoch in range(args.num_epochs):
    
        print("\nStart of epoch %d" % (epoch+1,))

        for step, (inputs, npts, gt) in enumerate(ds_train_iter):
            inputs = tf.convert_to_tensor(inputs, np.float32)
            gt = tf.convert_to_tensor(gt, np.float32)
            print(type(inputs),type(npts),type(gt))
            print(inputs.shape,sum(npts),gt.shape)
            with tf.GradientTape() as tape:

                coarse, fine = model((inputs, npts), training=True)  # Logits for this minibatch
                print(coarse.shape,fine.shape)

                """Total Loss Calculation"""
                gt_ds = gt[:, :coarse.shape[1], :]
                loss_coarse = earth_mover(coarse, gt_ds)
                loss_fine = chamfer(fine, gt)
                loss_value = loss_coarse + loss_fine

            print(len(model.trainable_weights),type(loss_value))
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Log every x batches.
            if step % 100 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * args.batch_size))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="data/complete/")
    parser.add_argument('--log_dir', default="/content/drive/MyDrive/pcn_altered_8192")
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_gt_points', type=int, default=16384)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--decay_steps', type=int, default=30000)
    parser.add_argument('--decay_rate', type=float, default=0.8)
    args = parser.parse_args()

    train(args)