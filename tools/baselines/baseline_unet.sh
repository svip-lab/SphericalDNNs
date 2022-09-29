CUDA_VISIBLE_DEVICES=1 python train_unet.py \
                         --batch-size 32\
                         -lr 3e-1\
                         -epochs 200\
                         --save-interval 10\
                         --exp-name 'ablation_unet-FTNT-node01'\
                         --snapshot-fname-dir 'ablation_unet_FTNT_node01_3e1_b32_2021_02_10'\
                         --snapshot-fname-prefix 'ablation_unet_FTNT_node01_3e1_b32_2021_02_10'
                         
                         
