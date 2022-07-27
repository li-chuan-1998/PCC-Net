import numpy as np
import tensorflow as tf
import open3d_util
import os

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, complete_dir, batch_size=32, shuffle=True):
        'Initialization'
        self.complete_dir = complete_dir
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        inputs, gt = self.gen_batch_inputs(list_IDs_temp)
        return inputs, gt

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def gen_batch_inputs(self, list_IDs_temp):
        def xform(dir):
            return dir.replace("complete", "partial")
        inputs, npts, gt = [], [], []
        for idx in list_IDs_temp:
            gt.append(open3d_util.read_pcd(os.path.join(self.complete_dir, idx)))
            temp_input = open3d_util.read_pcd(os.path.join(xform(self.complete_dir), xform(idx)))
            npts.append(len(temp_input))
            inputs.extend(temp_input)
        return (tf.convert_to_tensor([inputs], np.float32), npts), tf.convert_to_tensor(gt, np.float32)