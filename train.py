import tensorflow as tf
import argparse

from models.pcn import PCN
from lmdb_utils.lmdb_io import lmdb_dataflow

def train(args):
    # Data Pre-paration (utilising lmdb)
    df_train, num_train = lmdb_dataflow(
        args.lmdb_train, args.batch_size, args.num_input_points, args.num_gt_points, is_training=True)
    train_dataset = df_train.get_data()
    df_valid, num_valid = lmdb_dataflow(
        args.lmdb_valid, args.batch_size, args.num_input_points, args.num_gt_points, is_training=False)
    valid_dataset = df_valid.get_data()


    # Training & Validation
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args.base_lr,
        decay_steps=args.decay_steps,
        decay_rate=args.decay_rate,
        staircase=True)
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model = PCN()

    epochs = 2
    batch_size = 8
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (ids, inputs, npts, gt) in enumerate(train_dataset):

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                logits = model((ids, inputs, npts, gt), training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = model.loss

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
                print("Seen so far: %s samples" % ((step + 1) * batch_size))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_train', default='/content/drive/MyDrive/Normal Dataset/train.lmdb')
    parser.add_argument('--lmdb_valid', default='/content/drive/MyDrive/Normal Dataset/valid.lmdb')
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