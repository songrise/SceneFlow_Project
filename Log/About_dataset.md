# Week 2 Report


## Previous week issues.

- Normally, FlyingThings3D has annotation for scene flow (it is the dataset used for supervised training in FlowNet3D). But it is still possible that author remove annotation and train it in a self-supervised manner. The author did not state clearly how they pretrain the model. [Github issues on this topic](https://github.com/HimangiM/Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation/issues/11)
- About the output of model. It is the flow. More specifically, the input of the network $F$ is two point clouds $P,Q$, the output is flow $D$ (translational motion vectors for each point) , such that ideally $P+F(P,Q)=Q$. (ref. FlowNet3D)

- About the inference of "reversed flow". After forward flow we have our estimated point clouds for 2nd frame $Q^\prime{}=P+F(P,Q)$. Then the reversed flow is computed as $D^\prime = F(Q^\prime{},P)$. The Cycle consistency loss term is simply $||D+D^\prime ||$. (ref. FlowNet3D)

## This week

###  KITTI Scene Flow

- **Info**: originally designed for RGB stereo based method. **Raw data** of 3D Velodyne point clouds  are used in this paper (following FlowNet3D (Liu et al., CVPR 2019))
- **Makeup**: Original KITTI contains 200 scene, where 150 for train and 50 for testing. However, 50 test scene are not associated with raw point cloud data. Hence, only 150 scene (100 train, 50 test) are used in this project.
- metric: End Point Error (EPE), Acc (0.05): percentage
of scene flow prediction with an EPE < 0.05m or
relative error < 5% (also following FlowNet3D)
- preprocessing: ground point are removed.
- ** Details:**
- - Velodyne scans (i.e., the point clouds) are stored as floating point binaries, each point is stored with $(x, y, z)$ coordinate and reflectance value $(r)$, (however this project does not use $(r)$)
- - **Format**: each "scene" are associated with 3 attributes. `pos1; pos2; gt`, where `pos` are Cartesian coordinates of points in $R^3$, `gt` is the displacement vector, and pos1 + gt = pos2 . (unit?)
    - <img src="C:\Users\11385\AppData\Roaming\Typora\typora-user-images\image-20210615163526975.png" alt="image-20210615163526975" style="zoom: 50%;" />

### About the NuScenes dataset

- A larger dataset compared with KITTI, but no scene flow label is available for point clouds

- Actually, the authors preprocessed the NuScenes dataset but they did not release the processed dataset. 

- However, we can infer from the source code that the format of point clouds are similar as KITTI. *The major difference is that the point in this dataset has additional RGB color attributes.* (see `nuscenes_dataset_self_supervised_cycle.py` for more details)

## Source Codes

- The major part of codes are edited from pointnet++ and flownet3d.
- 

