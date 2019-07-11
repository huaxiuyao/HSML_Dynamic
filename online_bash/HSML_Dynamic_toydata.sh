#!/usr/bin/env bash
python main.py --datasource=mixture --metatrain_iterations=70000 --norm=None --update_batch_size=5 --update_batch_size_eval=10 --num_updates=5 --logdir=/media/huaxiuyao/My\ Passport1/ICML_logs_new/syncgroup_5shot_online_fix_bug --emb_loss_weight=0.01 --hidden_dim=40 --online_training=True --online_threshold=1.25 --cluster_layer_0=2

python main.py --datasource=mixture --metatrain_iterations=70000 --norm=None --update_batch_size=5 --update_batch_size_eval=10 --num_updates=5 --logdir=/media/huaxiuyao/My\ Passport1/ICML_logs_new/syncgroup_5shot_online_fix_bug --emb_loss_weight=0.01 --hidden_dim=40 --online_training=True --online_threshold=1.25 --cluster_layer_0=4 --train=False --test_set=True --test_epoch=69000 --num_test_task=4000
