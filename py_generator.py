import os


def xform(dir):
    return dir.replace("complete", "partial")

def get_file_paths(complete_dir, is_training=True):
    mid_pt = int(len(os.listdir(complete_dir))*7/9)
    start = 0 if is_training else mid_pt
    end = mid_pt if is_training else None
    return sorted(os.listdir(complete_dir))[start:end]

def gen_data(complete_dir, is_training=True):
    filenames = get_file_paths(complete_dir, is_training=is_training)
    
