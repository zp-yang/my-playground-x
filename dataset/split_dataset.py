import numpy as np
import glob
import os
import shutil

split_ratio = 0.90
seed = 42069
data_path = "/home/zp-yang/rosbag/datasets/"
out_dir_path = "/home/zp-yang/rosbag/split_dataset/"

train_out_dir = out_dir_path + "train/"
val_out_dir = out_dir_path + "val/"

for i in range(4):
    dataset_path = "{}dataset_{}".format(data_path, i)
    print("Current dataset path: {}".format(dataset_path))

    # only use annotated pictures
    data_anno = glob.glob(dataset_path+"/*.json")
    print("Number of annotated images: {}".format(len(data_anno)))

    data_fname = [os.path.splitext(fname)[0] for fname in data_anno]
    num_file = len(data_fname) 

    # randomly select training set and validation set
    rng = np.random.default_rng(seed=seed)
    train_data = rng.choice(data_fname, size=int(num_file*split_ratio), replace=False)
    val_data = [fname for fname in data_fname if fname not in train_data]
    print("number of train | val : {} | {}".format(len(train_data), len(val_data)))

    dataset_name = "dataset-{}-".format(i)
    for file in train_data:
        frame_name = dataset_name + os.path.basename(file)
        shutil.copy(file+".png", train_out_dir+frame_name+".png")
        shutil.copy(file+".json", train_out_dir+frame_name+".json")

    for file in val_data:
        frame_name = dataset_name + os.path.basename(file)
        shutil.copy(file+".png", val_out_dir+frame_name+".png")
        shutil.copy(file+".json", val_out_dir+frame_name+".json")