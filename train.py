import tensorflow as tf
import argparse
import time

from models.pcn import PCN
from dataloader import Dataloader
from loss_utils.utils import *

def get_alpha(step):
    rng = [10000, 20000, 50000]
    b = [0.01, 0.1, 0.5, 1.0]
    for ind, ele in enumerate(rng):
        if step < ele:
            return b[ind]
    return b[-1]

def train(args):
    # Data Pre-paration
    ds_train = Dataloader(complete_dir=args.data_path, is_training=True, batch_size=args.batch_size)
    ds_train_iter = iter(ds_train)
    ds_valid = Dataloader(complete_dir=args.data_path, is_training=False, batch_size=args.batch_size)
    ds_valid_iter = iter(ds_valid)

    # Model Initialization
    model = PCN()
    if args.restore:
        model.load_model(args.restore_path)
    else:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args.base_lr,
        decay_steps=args.decay_steps,
        decay_rate=args.decay_rate,
        staircase=True)
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer)
    
    
    print("Begin Training".center(100,"-"))
    total_step = 0
    train_start = time.time()
    for epoch in range(1,args.num_epochs+1):
        for step, (input, npts, gt) in enumerate(ds_train_iter):
            total_step+=1
            with tf.GradientTape() as tape:
                coarse, fine = model((input, npts), training=True)
                
                """Total Loss Calculation"""
                gt_ds = gt[:, :coarse.shape[1], :]
                loss_coarse = earth_mover(coarse, gt_ds)
                loss_fine = chamfer(fine, gt)
                loss_value = loss_coarse + loss_fine * get_alpha(total_step)

            grads = tape.gradient(loss_value, model.trainable_weights)
            model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if (total_step) % args.log_freq == 0:
                cur = (time.time() - train_start)/(total_step//args.log_freq)
                print(f"Epoch:{epoch:3d} Step:{total_step:8d} Loss:{float(loss_value):.6f} {cur/60:.3f}Min/100step")

        # Evaluation
        if epoch % args.eval_freq == 0:
            valid_start = time.time()
            print("Validating".center(100,"-"))
            total_loss = 0
            for step, (input, npts, gt) in enumerate(ds_valid_iter):
                coarse, fine = model((input, npts), training=False)
                """Total Loss Calculation"""
                gt_ds = gt[:, :coarse.shape[1], :]
                loss_coarse = earth_mover(coarse, gt_ds)
                loss_fine = chamfer(fine, gt)
                total_loss += (loss_coarse + loss_fine * get_alpha(total_step))
            val_end = time.time() - valid_start
            print(f"Epoch:{epoch:3d} Validation loss:{total_loss/(step+1)} {val_end/60:.3f}Min")

        # Save model's current weights
        if epoch % args.save_freq == 0:
            print(f"Model is savd at {args.save_path}".center(100,"-"))
            model.save(args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="data/complete/")
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--restore_path', default="/content/drive/MyDrive/pcn_tf_2/")
    parser.add_argument('--save_path', default="/content/drive/MyDrive/pcn_tf_2/")

    parser.add_argument('--log_freq', type=float, default=100)
    parser.add_argument('--eval_freq', type=float, default=1)
    parser.add_argument('--save_freq', type=float, default=5)

    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--decay_steps', type=int, default=5000)
    parser.add_argument('--decay_rate', type=float, default=0.8)
    args = parser.parse_args()

    train(args)