import tensorflow as tf
from loss_utils.utils import *

class PN_Conv1D_Layer(tf.keras.layers.Layer):
    def __init__(self, channels, momentum=0.5, name="pointnet_conv1d", **kwargs):
        super(PN_Conv1D_Layer, self).__init__(name=name, **kwargs)
        self.channels = channels
        self.momentum = momentum

    def build(self, input_shape: tf.Tensor):
        self.conv = tf.keras.layers.Conv1D(self.channels, 1, input_shape=input_shape)
        self.bn = tf.keras.layers.BatchNormalization(momentum=self.momentum)

    def call(self, inputs: tf.Tensor, training=None):
        return tf.nn.relu(self.bn(self.conv(inputs), training))

class Encoder_PN(tf.keras.layers.Layer):
    def __init__(self, name="encoder", **kwargs):
        super(Encoder_PN, self).__init__(name=name, **kwargs)

    def build(self, input_shape: tf.Tensor):
        self.conv_1 = PN_Conv1D_Layer(128)
        self.conv_2 = PN_Conv1D_Layer(256)
        self.conv_3 = PN_Conv1D_Layer(512)
        self.conv_4 = PN_Conv1D_Layer(1024)
    
    def call(self, inputs):
        inputs, npts= inputs
        # 1st layer of pointnet
        features = self.conv_2(self.conv_1(inputs))
        features_global = self.point_unpool(self.point_maxpool(features, npts, keepdims=True), npts)
        features = tf.concat([features, features_global], axis=2)

        # 2nd layer of pointnet
        features = self.point_maxpool(self.conv_4(self.conv_3(features)), npts)
        return features

    def point_maxpool(self, inputs, npts, keepdims=False):
        outputs = [tf.reduce_max(f, axis=1, keepdims=keepdims)
            for f in tf.split(inputs, npts, axis=1)]
        return tf.concat(outputs, axis=0)

    def point_unpool(self, inputs, npts):
        inputs = tf.split(inputs, len(npts), axis=0)
        outputs = [tf.tile(f, [1, npts[i], 1]) for i,f in enumerate(inputs)]
        return tf.concat(outputs, axis=1)

class Coarse_Layer(tf.keras.layers.Layer):
    def __init__(self, num_coarse=1024, name="coarse_layer", **kwargs):
        super(Coarse_Layer, self).__init__(name=name, **kwargs)
        self.num_coarse = num_coarse

    def build(self, input_shape: tf.Tensor):
        self.dense_1 = tf.keras.layers.Dense(1024)
        self.dense_1 = tf.keras.layers.Dense(1024)
        self.dense_3 = tf.keras.layers.Dense(self.num_coarse*3)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = self.dense_3(self.dense_1(self.dense_1(inputs)))
        return tf.reshape(inputs, [-1, self.num_coarse, 3])

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_coarse=1024, grid_scale=0.05, grid_size=4, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.num_coarse = num_coarse
        self.grid_scale = grid_scale
        self.grid_size = grid_size
        self.num_fine = self.grid_size ** 2 * self.num_coarse

    def build(self, input_shape: tf.Tensor):
        self.coarse_layer = Coarse_Layer()
        self.dense_1 = tf.keras.layers.Dense(512)
        self.dense_2 = tf.keras.layers.Dense(512)
        self.dense_3 = tf.keras.layers.Dense(3)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        partial = inputs
        coarse = self.coarse_layer(partial)
        
        x = tf.raw_ops.LinSpace(start=-self.grid_scale, stop=self.grid_scale, num=self.grid_size)
        y = tf.raw_ops.LinSpace(start=-self.grid_scale, stop=self.grid_scale, num=self.grid_size)
        grid = tf.meshgrid(x, y)
        grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
        grid_feat = tf.tile(grid, [len(partial), self.num_coarse, 1])

        point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        point_feat = tf.reshape(point_feat, [-1, self.num_fine, 3])

        global_feat = tf.tile(tf.expand_dims(partial, 1), [1, self.num_fine, 1])

        feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)

        center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        center = tf.reshape(center, [-1, self.num_fine, 3])
        fine = self.dense_3(self.dense_2(self.dense_1(feat))) + center

        return coarse, fine


loss_tracker = tf.keras.metrics.Mean(name="loss")

class PCN(tf.keras.Model):
    def __init__(self, name="pcn_model", **kwargs):
        super(PCN, self).__init__(name=name, **kwargs)
        self.encoder = Encoder_PN()
        self.decoder = Decoder()
        self.step = 0

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs, npts = inputs
        features = self.encoder((inputs, npts))
        coarse, fine = self.decoder(features)
        return coarse, fine

    def train_step(self, data):
        self.step+=1
        inputs, gt = data

        with tf.GradientTape() as tape:
            coarse, fine = self(inputs, training=True)

            """Total Loss Calculation"""
            gt_ds = gt[:, :coarse.shape[1], :]
            loss_coarse = earth_mover(coarse, gt_ds)
            loss_fine = chamfer(fine, gt)
            loss_value = loss_coarse + loss_fine * self.get_alpha(self.step)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss_value, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        loss_tracker.update_state(loss_value)
        return {"loss": loss_tracker.result()}

    def test_step(self, data):
        inputs, gt = data
        coarse, fine = self(inputs, training=False)

        gt_ds = gt[:, :coarse.shape[1], :]
        loss_coarse = earth_mover(coarse, gt_ds)
        loss_fine = chamfer(fine, gt)
        loss_value = loss_coarse + loss_fine * self.get_alpha(self.step)
        loss_tracker.update_state(loss_value)
        return {"loss": loss_tracker.result()}

    @property
    def metrics(self):
        return [loss_tracker]

    def get_alpha(step):
        rng = [10000, 20000, 50000]
        b = [0.01, 0.1, 0.5, 1.0]
        for ind, ele in enumerate(rng):
            if step < ele:
                return b[ind]
        return b[-1]
