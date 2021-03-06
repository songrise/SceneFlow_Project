


python train_1nn_cycle_nuscenes.py \
    --model model_concat_upsa_1nn_cycle_nuscenes \
    --data data_preprocessing/nuscenes_trainval_rgb_pkl_600_full \
    --log_dir logs/log_train_cycle_nuscenes \
    --num_point 2048 \
    --batch_size 8 \
    --radius 5 \
    --layer pointnet \
    --cache_size 0 \
    --gpu 3 \
    --learning_rate 0.001 \
    --dataset nuscenes_dataset_self_supervised_cycle \
    --num_frames 2 \
    --fine_tune \
    --model_path pretrained_models/log_train_pretrained/model.ckpt \
    --max_epoch 10000 \
    --flip_prob 0.5
