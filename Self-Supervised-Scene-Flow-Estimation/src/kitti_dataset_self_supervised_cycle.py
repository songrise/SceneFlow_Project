import os
import os.path
import json
import numpy as np
import sys
import pickle
import glob
import random
# import mayavi.mlab as mlab


class SceneflowDataset():
    def __init__(self, root='./data_preprocessing/kitti_self_supervised_flow',
                 cache_size=30000, npoints=2048, train=True,
                 softmax_dist=False, num_frames=3, flip_prob=0,
                 sample_start_idx=-1):
        # !Re: number of points for each point clouds (i.e. subsampling)
        self.npoints = npoints
        self.train = train
        self.root = root
        if self.train:
            self.datapath = glob.glob(os.path.join(self.root, 'train/*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root, 'test/*.npz'))
        self.cache = {}
        self.cache_size = cache_size
        self.softmax_dist = softmax_dist
        # !Re: 2, actually the processed dataset contains 2 frame only
        self.num_frames = num_frames
        #!Re: probability to flip the scene.
        self.flip_prob = flip_prob
        self.sample_start_idx = sample_start_idx

    def __getitem__(self, index):
        if index in self.cache:
            pos_list, color_list = self.cache[index]
        else:
            fn = self.datapath[index]
            pc_np_list = np.load(fn)
            pc_list = []  # !Re: a list of list for two clouds
            pc_list.append(pc_np_list['pos1'])
            pc_list.append(pc_np_list['pos2'])

            #! Re: randomly choose one index of frame (0 or 1) as start index
            #! when there are two frames, this is temporal flip augmentation
            start_idx = np.random.choice(np.arange(len(pc_list)-self.num_frames+1),
                                         size=1)[0]
            # start_idx = 0
            pos_list = []
            color_list = []
            min_length = np.min([len(x) for x in pc_list])
            # print (min_length, min_length-self.npoints+1)
            if self.sample_start_idx == -1:
                sample_start_idx = np.random.choice(min_length-self.npoints+1,
                                                    size=1)[0]
            else:
                sample_start_idx = self.sample_start_idx

            #!Re: the indecies for points that is subsampled in both frame
            sample_idx = np.arange(sample_start_idx,
                                   sample_start_idx+self.npoints)

            for frame_idx in range(start_idx, start_idx + self.num_frames):
                data = pc_list[frame_idx]  # num_point x 4
                pos = data[sample_idx, :3]  # !Re: xyz

                #! Re: color is all zeros. Although kitti point clouds does not have rgb.
                #! purpose of this is that to be compatible with nuscene dataset,
                #! see https://github.com/HimangiM/Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation/blob/0a3350843de1ed769e69c3be17eb70db32ca6881/src/train_1nn_cycle_nuscenes.py#L304

                color = np.zeros((len(sample_idx), 3))

                pos_list.append(pos)
                color_list.append(color)

            prob = random.uniform(0, 1)
            if prob < self.flip_prob:
                #! Re: Invariance under transformations for point clouds ?
                pos_list = pos_list[::-1]
                color_list = color_list[::-1]  # ! why color also flip

            if len(self.cache) < self.cache_size:
                #! Re store in memory
                self.cache[index] = (pos_list, color_list)

        return np.array(pos_list), np.array(color_list)

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    d = SceneflowDataset(
        root="/home/songrise/Desktop/Summer_Research/Self-Supervised-Scene-Flow-Estimation/data_preprocessing/kitti_self_supervised_flow", npoints=2048, train=True)
    print('Len of dataset:', len(d))
    # print(d[10])
    bsize = 4
    # idxs =
    for i in range(bsize):
        # ipdb.set_trace()
        # if dataset[0] == None:
        #     print (i, bsize)
        pos, color = d[idxs[i + start_idx]]
