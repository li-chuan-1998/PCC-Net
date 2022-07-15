import tensorflow as tf

from models.pcn import PCN

# Data Pre-paration (utilising lmdb)
training_dataset = None
valid_dataset = None


# Training & Validation
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=3000,
    decay_rate=0.9,
    staircase=True)

model = PCN()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))

for idx, data in enumerate(training_dataset):
    pass