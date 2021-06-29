import sys
import numpy as np
import open3d as o3d
import os

BASE_DIR = 'Self-Supervised-Scene-Flow-Estimation/src'
sys.path.append(BASE_DIR)


def stat(a,b,g):
    print("max of a ",np.max(a[:,0]), np.max(a[:,1]),np.max(a[:,2]))
    print("max of gt ",np.max(g[:,0]), np.max(g[:,1]),np.max(g[:,2]))
    print("mean of gt",np.mean(g[:,0]), np.mean(g[:,1]),np.mean(g[:,2]))


_dir = '/home/songrise/Desktop/SceneFlow_Project/Self-Supervised-Scene-Flow-Estimation/data_preprocessing/kitti_self_supervised_flow/train'
file_name = '000000.npz'
point_cloud = np.load(os.path.join(_dir, file_name))
# point_cloud = np.random.choice(point_cloud,size = len(point_cloud)*RAND_DOWN_SAMPLE_RATE)
print(point_cloud['pos1'].shape)
pc1 = point_cloud['pos1']
pc2 = point_cloud['pos2']
gt = point_cloud['gt']
gt_flow = gt+pc1

stat(pc1,pc2,gt)


pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(pc1)
pcd1.paint_uniform_color([1, 0, 0])  # ! Re: red

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(pc2)
pcd2.paint_uniform_color([0, 1, 0])  # ! Re: green

pcd3 = o3d.geometry.PointCloud()
pcd3.points = o3d.utility.Vector3dVector(gt_flow)
pcd3.paint_uniform_color([0, 0, 1])  # ! Re: blue

o3d.visualization.draw_geometries([pcd1, pcd2,pcd3])