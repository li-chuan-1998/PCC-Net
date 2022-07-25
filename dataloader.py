import os
import open3d_util
import numpy as np
import tensorflow as tf

def xform(dir):
    return dir.replace("complete", "partial")

def get_file_paths(complete_dir, is_training=True):
    mid_pt = int(len(os.listdir(complete_dir))*7/9)
    start = 0 if is_training else mid_pt
    end = mid_pt if is_training else None
    return sorted(os.listdir(complete_dir))[start:end]


class Dataloader:
    def __init__(self, complete_dir, is_training=True, batch_size=32):
        self.complete_dir = complete_dir
        self.batch_size = batch_size
        self.list_pcd_path = get_file_paths(complete_dir, is_training=is_training)
        self.shuffled_idx = self.split_idx()
        self.counter = 0

    def gen_batch_inputs(self, index):
        inputs, npts, gt = [], [], []
        for idx in self.shuffled_idx[index]:
            cur_pcd_path = self.list_pcd_path[idx]
            gt.append(open3d_util.read_pcd(os.path.join(self.complete_dir, cur_pcd_path)))
            temp_input = open3d_util.read_pcd(os.path.join(xform(self.complete_dir), xform(cur_pcd_path)))
            inputs.extend(temp_input)
            npts.append(len(temp_input))
        return tf.convert_to_tensor([inputs], np.float32), npts, tf.convert_to_tensor(gt, np.float32)

    def split_idx(self):
        rnd_idx = np.random.permutation(len(self.list_pcd_path)).tolist()
        batch_ids = []
        for i in range(int(np.ceil(len(rnd_idx)/self.batch_size))):
            start = i*self.batch_size
            end = None if start+self.batch_size >= len(rnd_idx) else start+self.batch_size
            batch_ids.append(rnd_idx[start:end])
        return batch_ids
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.counter < len(self.shuffled_idx):
            self.counter+=1
            return self.gen_batch_inputs(self.counter-1)
        else:
            self.counter = 0
            self.shuffled_idx = self.split_idx()
            raise StopIteration
