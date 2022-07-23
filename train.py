import tensorflow as tf
import numpy as np
import argparse
import os

from models.pcn import PCN
from dataloader import Dataloader

def train(args):
    # Data Pre-paration
    ds_train = Dataloader(complete_dir=args.data_path, is_training=True, batch_size=args.batch_size)
    ds_train_iter = iter(ds_train)
    ds_valid = Dataloader(complete_dir="data/complete/", is_training=False, batch_size=8)
    ds_valid_iter = iter(ds_valid)

    # Training & Validation
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args.base_lr,
        decay_steps=args.decay_steps,
        decay_rate=args.decay_rate,
        staircase=True)
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model = PCN()

    if args.restore:
        latest = tf.train.latest_checkpoint(args.checkpoint_dir)
        model.load_weights(latest)
    
    print("------------------Training Begins------------------")
    total_step = 0
    for epoch in range(1,args.num_epochs+1):
        for step, batch_data in enumerate(ds_train_iter):
            total_step+=1
            with tf.GradientTape() as tape:
                coarse, fine = model(batch_data, training=True)
                loss_value = sum(model.losses)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if (step+1) % args.log_freq == 0:
                print(f"Epoch: {epoch} Total Step: {total_step} Lr: {float(model.optimizer.lr)} Training loss: {float(loss_value)}")

        # Evaluate
        if epoch % args.eval_freq == 0:
            total_loss = 0
            for step, batch_data in enumerate(ds_valid_iter):
                coarse, fine = model(batch_data, training=False)
                total_loss += float(sum(model.losses))
            print(f"Epoch: {epoch} Validation loss: {total_loss/(step+1)}")

        # Save model's current weights in every x epochs
        if epoch % args.save_freq == 0:
            model.save_weights(os.path.join(args.checkpoint_dir, "cp-{epoch:06d}.ckpt".format(epoch=epoch)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="data/complete/")
    parser.add_argument('--checkpoint_dir', default="/content/drive/MyDrive/pcn_tf_2/")
    parser.add_argument('--restore', action='store_true', default=True)
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