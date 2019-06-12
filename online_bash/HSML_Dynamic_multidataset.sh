#!/usr/bin/env bash
python main.py --datasource=multidataset --metatrain_iterations=40000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=/media/huaxiuyao/My\ Passport1/ICML_logs_new/multidataset_1shot_online_fix_bug/ --num_filters=32 --max_pool=True --hidden_dim=128 --emb_loss_weight=0.01 --online_training=True --online_threshold=0.85 --cluster_layer_0=2

python main.py --datasource=multidataset --metatrain_iterations=40000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=/media/huaxiuyao/My\ Passport1/ICML_logs_new/multidataset_1shot_online_fix_bug/ --num_filters=32 --max_pool=True --hidden_dim=128 --emb_loss_weight=0.01 --online_training=True --online_threshold=0.85 --cluster_layer_0=4 --train=False --test_set=True --test_dataset=2 --test_epoch=39000