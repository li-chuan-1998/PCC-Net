import tensorflow as tf
import argparse

from models.pcn import PCN
from loss_utils.emd import *
from loss_utils.cd import *

def train(args):
    # Data Pre-paration (utilising lmdb)
    train_dataset = None


    # Training & Validation
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args.base_lr,
        decay_steps=args.decay_steps,
        decay_rate=args.decay_rate,
        staircase=True)
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model = PCN()

    epochs = 2
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (inputs, npts, gt) in enumerate(train_dataset):

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                coarse, fine = model((inputs, npts), training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = sum(model.losses)

                # Total Loss Calculation
                gt_ds = gt[:, :coarse.shape[1], :]
                loss_coarse = getEMD(coarse.numpy(), gt_ds.numpy())
                loss_fine = chamfer_distance_tf(fine, gt)/2
                total_loss = loss_coarse + loss_fine
                loss_value += total_loss 


            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * args.batch_size))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_train', default='/home/wrs/pcn/dataset_v3/train.lmdb')
    parser.add_argument('--lmdb_valid', default='/home/wrs/pcn/dataset_v3/valid.lmdb')
    parser.add_argument('--log_dir', default="/content/drive/MyDrive/pcn_altered_8192")
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_input_points', type=int, default=8192)
    parser.add_argument('--num_gt_points', type=int, default=16384)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--decay_steps', type=int, default=30000)
    parser.add_argument('--decay_rate', type=float, default=0.8)
    args = parser.parse_args()

    train(args)