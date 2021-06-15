# About the dataset

## KITTI Scene Flow
- **Info**: originally designed for RGB stereo based method. **Raw data** of 3D Velodyne point clouds  are used in this paper (following FlowNet3D (Liu et al., CVPR 2019))
- **Makeup**: Original KITTI contains 200 scene, where 150 for train and 50 for testing. However, 50 test scene are not associated with raw point cloud data. Hence, only 150 scene (100 train, 50 test) are used in this project.
- metric: End Point Error (EPE), Acc (0.05): percentage
of scene flow prediction with an EPE < 0.05m or
relative error < 5% (also following FlowNet3D)
- preprocessing: ground point are removed.
-** Details:**
- - Velodyne scans (i.e., the point clouds) are stored as floating point binaries, each point is stored with $(x, y, z)$ coordinate and reflectance value $(r)$, (however this project does not use $(r)$)
- - **Format**: each "scene" are associated with 3 attributes. `pos1; pos2; gt`, where `pos` are Cartesian coordinates of points in $R^3$, `gt` is the displacement vector, and pos1 + gt = pos2 . (unit?)<img src="C:\Users\11385\AppData\Roaming\Typora\typora-user-images\image-20210615163526975.png" alt="image-20210615163526975" style="zoom: 50%;" />

