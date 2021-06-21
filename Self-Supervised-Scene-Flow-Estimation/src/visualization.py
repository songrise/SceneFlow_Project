import numpy as np
import open3d as o3d
import os
import argparse
import math
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import glob

BASE_DIR = 'Self-Supervised-Scene-Flow-Estimation/src'
sys.path.append(BASE_DIR)
import pickle
import pdb
import utils.tf_util
from utils.pointnet_util import *
from tf_grouping import query_ball_point, group_point, knn_point
from scipy.spatial import distance_matrix

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!Re: some tutorial on open3d http://www.open3d.org/docs/release/tutorial

#!Re a example of subsampling a patch of 2048 points.
SUBSAMPLE = False
#!Re: use model to infer the forward flow `gt`
INFERENCE = True

_dir = 'data_preprocessing/kitti_self_supervised_flow/train'
file_name = '000000.npz'
point_cloud = np.load(os.path.join(_dir, file_name))
if not SUBSAMPLE:
    pc1 = point_cloud['pos1']
    pc2 = point_cloud['pos2']
    gt_flow = point_cloud['gt'] + pc1
else:
    pc1 = point_cloud['pos1'][:2048, :3]
    pc2 = point_cloud['pos2'][:2048, :3]
    #!Re: ideally, pc2 == gt_flow
    gt_flow = point_cloud['gt'][:2048, :3] + pc1

print('Point Cloud 1 shape:{}, Point Cloud 2 shape:{}, Flow Shape:{}'.format(
    pc1.shape, pc2.shape, gt_flow.shape))
if not INFERENCE:
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1)
    pcd1.paint_uniform_color([1, 0, 0])  # ! Re: red

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pc2)
    pcd2.paint_uniform_color([0, 1, 0])  # ! Re: green

    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(gt_flow)
    pcd3.paint_uniform_color([0, 0, 1])  # ! Re: blue

    o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)

    len_cloud = 100000

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use [default: GPU 0]')
    parser.add_argument('--model', default='model_concat_upsa',
                        help='Model name [default: model_concat_upsa]')
    parser.add_argument('--data', default='data_preprocessing/data_processed_maxcut_35_20k_2k_8192',
                        help='Dataset directory [default: /data_preprocessing/data_processed_maxcut_35_20k_2k_8192]')
    parser.add_argument('--model_path', default='log_train_pretrained/model.ckpt',
                        help='model checkpoint file path [default: log_train/model.ckpt]')
    parser.add_argument('--num_point', type=int, default=2048,
                        help='Point Number [default: 2048]')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch Size during training [default: 16]')
    parser.add_argument('--num_frames', type=int, default=2,
                        help='Number of frames to run cycle')
    parser.add_argument('--decay_step', type=int, default=200000,
                        help='Decay step for lr decay [default: 200000]')
    parser.add_argument('--decay_rate', type=float, default=0.7,
                        help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--radius', type=float, default=5.0,
                        help='Radius of flow embedding layer')
    parser.add_argument('--layer', type=str, default='pointnet',
                        help='Last layer for upconv')
    parser.add_argument('--knn', action='store_true',
                        help='knn or query ball point')
    parser.add_argument('--flow', type=str, default='default',
                        help='flow embedding module type')
    parser.add_argument('--kitti_dataset', default='data_preprocessing/kitti_self_supervised_flow',
                        help='Dataset directory [default: /data_preprocessing/kitti_self_supervised_flow]')

    FLAGS = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

    BATCH_SIZE = FLAGS.batch_size
    NUM_POINT = FLAGS.num_point
    DATA = "data_preprocessing/kitti_self_supervised_flow"
    GPU_INDEX = FLAGS.gpu
    NUM_FRAMES = FLAGS.num_frames
    DECAY_STEP = FLAGS.decay_step
    DECAY_RATE = FLAGS.decay_rate
    RADIUS = FLAGS.radius

    print(FLAGS)

    MODEL = importlib.import_module(FLAGS.model)  # import network module
    MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model + '.py')
    MODEL_PATH = FLAGS.model_path
    LAYER = FLAGS.layer
    KNN = FLAGS.knn
    FLOW_MODULE = FLAGS.flow
    KITTI_DATASET = FLAGS.kitti_dataset

    BN_INIT_DECAY = 0.5
    BN_DECAY_DECAY_RATE = 0.5
    BN_DECAY_DECAY_STEP = float(DECAY_STEP)
    BN_DECAY_CLIP = 0.99






    def scene_flow_EPE_np(pred, labels, mask):
        error = np.sqrt(np.sum((pred - labels) ** 2, 2) + 1e-20)

        gtflow_len = np.sqrt(np.sum(labels * labels, 2) + 1e-20)  # B,N
        acc1 = np.sum(np.logical_or((error <= 0.05) * mask, (error / gtflow_len <= 0.05) * mask), axis=1)
        acc2 = np.sum(np.logical_or((error <= 0.1) * mask, (error / gtflow_len <= 0.1) * mask), axis=1)

        mask_sum = np.sum(mask, 1)
        acc1 = acc1[mask_sum > 0] / mask_sum[mask_sum > 0]
        acc1 = np.sum(acc1)
        acc2 = acc2[mask_sum > 0] / mask_sum[mask_sum > 0]
        acc2 = np.sum(acc2)

        EPE = np.sum(error * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
        EPE = np.sum(EPE)
        return EPE, acc1, acc2, error, gtflow_len


    def get_bn_decay(batch):
        bn_momentum = tf.train.exponential_decay(
            BN_INIT_DECAY,
            batch * BATCH_SIZE,
            BN_DECAY_DECAY_STEP,
            BN_DECAY_DECAY_RATE,
            staircase=True)
        bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
        return bn_decay


    def return_dist_threshold(pc1, threshold=1.0):
        all_dist_count = []
        diff_matrix = distance_matrix(pc1, pc1)
        r, c = np.where(diff_matrix <= threshold)
        _, counts_elements = np.unique(r, return_counts=True)
        for i in counts_elements:
            all_dist_count.append(i)

        return all_dist_count


    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl, masks_pl = MODEL.placeholder_inputs(
                None, NUM_POINT)

            is_training_pl = tf.placeholder(tf.bool, shape=())

            batch = tf.Variable(0)  # batch = 0
            bn_decay = get_bn_decay(batch)  # bn_decay = 0.5
            print("--- Get model and loss")
            # Get model and loss
            pred, end_points = MODEL.get_model(RADIUS, LAYER, pointclouds_pl, is_training_pl, bn_decay=bn_decay,
                                               knn=KNN, flow_module=FLOW_MODULE)

            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        saver.restore(sess, MODEL_PATH)

        ops = {'pointclouds_pl': pointclouds_pl,
               'label': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'end_points': end_points}
        is_training = False
        #! Re: modified here.


        l_error_all = []

        num_frame = 40
        epe_total = 0
        epe_count = 0
        sample_count = 0

        x  = np.load('data_preprocessing/kitti_self_supervised_flow/train/'+file_name)
        all_pred = []
        all_label = []
        all_points = []
        batch_label = []
        batch_data = []
        batch_mask = []
        ref_pc = x['pos1'][:, :3]
        ref_center = np.mean(ref_pc, 0)
        print(len(x['pos1']), len(x['pos2']))
        for i in range(0, len_cloud, 2048):
            if i + 2048 < len(x['pos1']) and i + 2048 < len(x['pos2']):
                pc1 = x['pos1'][i:i + 2048, :3]
                pc2 = x['pos2'][i:i + 2048, :3]
                gt = x['gt'][i:i + 2048, :3]
                pc1 = pc1 - ref_center
                pc2 = pc2 - ref_center
                batch_data.append(np.concatenate([np.concatenate([pc1,
                                                                  pc2], axis=0),
                                                  np.zeros((4096, 3))], axis=1))  # 4096, 6
                batch_label.append(gt)
        batch_data = np.array(batch_data)
        batch_label = np.array(batch_label)
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['is_training_pl']: is_training, }
        pred_val, end_points_val = sess.run([ops['pred'], ops['end_points']], feed_dict=feed_dict)

        #!Re: visualization here
        pc1 = point_cloud['pos1'][:2048, :3]
        pc2 = point_cloud['pos2'][:2048, :3]
        #!Re since author hard-coded 2048, we follow this and use first sampling patch for visualization.
        gt_flow = pred_val[0] + pc1 #! predicted second frame

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pc1)
        pcd1.paint_uniform_color([1, 0, 0])  # ! Re: red

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pc2)
        pcd2.paint_uniform_color([0, 1, 0])  # ! Re: green

        pcd3 = o3d.geometry.PointCloud()
        pcd3.points = o3d.utility.Vector3dVector(gt_flow)
        pcd3.paint_uniform_color([0, 0, 1])  # ! Re: blue

        o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])

