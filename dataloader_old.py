import os
import open3d_util
import numpy as np
import tensorflow as tf

def xform(dir):
    return dir.replace("complete", "partial")

def get_file_paths(complete_dir, training=True):
    mid_pt = int(len(os.listdir(complete_dir))*7/9)
    start = 0 if training else mid_pt
    end = mid_pt if training else None
    return sorted(os.listdir(complete_dir))[start:end]


class Dataloader:
    def __init__(self, complete_dir, is_training=True, batch_size=8):
        self.complete_dir = complete_dir
        self.batch_size = batch_size
        self.is_training = is_training
        self.data = self.get_pcds_np()
        self.shuffled_idx = self.split_idx()
        self.counter = 0
    
    def get_pcds_np(self):
        partial, npts, complete = [], [], []
        cnt = 0
        for pcd_path in get_file_paths(self.complete_dir, self.is_training):
            complete.append(open3d_util.read_pcd(os.path.join(self.complete_dir, pcd_path)))
            partial.append(open3d_util.read_pcd(os.path.join(xform(self.complete_dir), xform(pcd_path))))
            npts.append(len(partial[-1]))
            cnt+=1
            if cnt%500 == 0: print("-",end=" ")
        print("\nDone")
        return partial, npts, complete
    
    def split_idx(self):
        rnd_idx = np.random.permutation(len(self.data[0])).tolist()
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
            partial, npts, complete = [], [], []
            for idx in self.shuffled_idx[self.counter]:
                partial.extend(self.data[0][idx])
                npts.append(self.data[1][idx])
                complete.append(self.data[2][idx])
            self.counter+=1
            # return tf.convert_to_tensor(np.asarray([partial]), np.float32), npts, tf.convert_to_tensor(np.asarray(complete), np.float32)
            return np.asarray([partial]), npts, np.asarray(complete)
        else:
            self.counter = 0
            self.shuffled_idx = self.split_idx()
            raise StopIteration

    def split_to_batch(self):
        batch_i, batch_n, batch_g = [], [], []
        while self.counter < len(self.shuffled_idx):
            temp_i, temp_n, temp_g = self.__next__()
            batch_i.append(temp_i)
            batch_n.append(temp_n)
            batch_g.append(temp_g)
        return batch_i, batch_n, batch_g 

