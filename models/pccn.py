import tensorflow as tf

class PointNet(tf.keras.layers.Layer):
    def __init__(self, input_shape, out_dim):
        super(PointNet, self).__init__()
        self.bs = input_shape[0]
        self.num_pts = input_shape[1]
        self.conv_1 = tf.keras.layers.Conv2D(64, [1,3], activation='relu')
        self.conv_2 = tf.keras.layers.Conv2D(64, [1,1], activation='relu')
        self.conv_3 = tf.keras.layers.Conv2D(128, [1,1], activation='relu')
        self.conv_4 = tf.keras.layers.Conv2D(512, [1,1], activation='relu')
        self.conv_5 = tf.keras.layers.Conv2D(1024, [1,1], activation='relu')
        self.max_pool2d = tf.keras.layers.MaxPool2D(pool_size=[None, 1], strides=(2, 2), padding='valid')
        self.dense_1 = tf.keras.layers.Dense(1024,activation='relu')
        self.dense_2 = tf.keras.layers.Dense(512,activation='relu')
        self.dense_3 = tf.keras.layers.Dense(out_dim, activation='relu')
    
    def call(self, input_tensor):
        x = tf.expand_dims(input_tensor, axis=2)
        x = self.max_pool2d(self.conv_5(self.conv_4(self.conv_3(self.conv_2(self.conv_1(input_tensor))))))
        x = tf.reshape(x, [self.bs, -1])
        x = self.dense_3(self.dense_2(self.dense_1(x)))
        return x

class MLP(tf.keras.layers.Layer):
    def __init__(self):
        super(MLP, self).__init__()
        self.dense_1 = tf.keras.layers.Dense(512,activation='relu')
        self.dense_2 = tf.keras.layers.Dense(1024,activation='relu')
        self.dense_3 = tf.keras.layers.Dense(2048, activation='relu')
        self.dense_4 = tf.keras.layers.Dense(4096, activation='relu')
        self.dense_5 = tf.keras.layers.Dense(8192, activation='relu')
    
    def call(self, input_tensor):
        x = self.dense_5(self.dense_4(self.dense_3(self.dense_2(self.dense_1(input_tensor)))))
        return x


class PCC_Net(tf.keras.Model):
    def __init__(self):
        pass

"""
    def encode(self, x, training):
        encoded = self.encoder(x, training=training)
        mu = encoded[:, :self.k]
        log_sigma = encoded[:, self.k:]

        return mu, log_sigma
    
    def decode(self, z, training):
        x = self.decoder(z, training=training)

        return x
    
    def reparam(self, eps, mu, log_sigma):
        sigma = tf.exp(log_sigma)
        z = tf.sqrt(sigma) * eps + mu

        return z
    
    def encoder_loss(self, mu, log_sigma):
        sigma = tf.exp(log_sigma)
        loss = (1/2) * (
            tf.reduce_sum(sigma, axis=-1, keepdims=True) + \
            tf.reduce_sum(mu**2, axis=-1, keepdims=True) - \
            self.k - \
            tf.reduce_sum(log_sigma, axis=-1, keepdims=True)
        )

        return loss
    
    def decoder_loss(self, x, f_z):
        loss = tf.reduce_sum((x - f_z)**2, axis=-1, keepdims=True)

        return loss

    def encode_loss(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar
"""